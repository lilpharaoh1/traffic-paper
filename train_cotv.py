import os.path
import random

import ray
from ray import tune
import argparse
import configparser
import sys
import logging
from ray.tune.registry import register_env
from ray.tune import PlacementGroupFactory
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.rllib.algorithms.registry import get_algorithm_class

from scenario.scen_retrieve import SumoScenario
from envs.env_register import register_env_gym
from policies import PolicyConfig

import logging

# ----- Customised functions (multiagent) -----

def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    # to customise different agent type
    if '.' in agent_id:
        return "policy_0"  # CAV
    else:
        return "policy_1"  # TL


def parse_args(args):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Parse argument used when running a DRL training for traffic control.",
        epilog="python train.py EXP_CONFIG")

    # ----required input parameters----
    parser.add_argument(
        '--exp_config', type=str, default='cotv_config.ini',
        help='Name of the experiment configuration file, as located in exp_configs.')

    # ----optional input parameters----
    parser.add_argument(
        '--log_level', type=str, default='ERROR',
        help='Level setting for logging to track running status.'
    )

    return parser.parse_known_args(args)[0]


def main(args):
    args = parse_args(args)
    logging.basicConfig(level=args.log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.info(f"DRL training with the following CLI args: {args}")
    # ray.init(local_mode=True, log_to_driver=True, logging_level=logging.DEBUG)

    # import experiment configuration
    config_file = args.exp_config
    config = configparser.ConfigParser()
    config.read(os.path.join('./exp_configs', config_file))
    if not config:
        logger.error(f"Unable to find the experiment configuration {config_file} in exp_configs")
    config.set('TRAIN_CONFIG', 'log_level', args.log_level)

    # 1. process SUMO scenario files to get info, required in make env
    scenario = SumoScenario(config['SCEN_CONFIG'])
    logger.info(f"The scenario cfg file is {scenario.cfg_file_path}.")

    # 2. register env and make env in OpenAIgym, then register_env in Ray
    this_env = __import__("envs", fromlist=[config.get('TRAIN_CONFIG', 'env')])
    if hasattr(this_env, config.get('TRAIN_CONFIG', 'env')):
        this_env = getattr(this_env, config.get('TRAIN_CONFIG', 'env'))
    this_env_register, env_name = register_env_gym(this_env, scenario, config['SUMO_CONFIG'], config['CONTROL_CONFIG'],
                                                   config['TRAIN_CONFIG'])
    register_env(env_name, this_env_register)
    this_env = this_env(scenario, config['SUMO_CONFIG'], config['CONTROL_CONFIG'], config['TRAIN_CONFIG'])

    # 3. set DRL algorithm/model
    policy_config = PolicyConfig(env_name, config['ALG_CONFIG'], config['TRAIN_CONFIG'], config['MODEL_CONFIG'])
    
    # 4. make spec
    # policy_class = config['POLICY_SPEC'].get('policy_class', fallback=None)
    # if policy_class:
    #     policy_class = get_algorithm_class(policy_class)
    # obs_space = getattr(this_env, config['POLICY_SPEC'].get('observation_space'), None)
    # act_space = getattr(this_env, config['POLICY_SPEC'].get('action_space'), None)
    # spec = SingleAgentRLModuleSpec(policy_class, obs_space, act_space) 
    # policy_config.config_to_ray.update({'rl_module_spec': spec})
    policy_config.config_to_ray.update({'disable_env_checking': True})  # to avoid checking non-override default obs_space...

    # for section in config.sections():
    #     if 'policySpec' in section:
    #         policy_class = config.get(section, 'policy_class', fallback=None)
    #         if policy_class:
    #             policy_class = get_algorithm_class(policy_class)
    #         obs_space = getattr(this_env, config.get(section, 'observation_space'), None)
    #         act_space = getattr(this_env, config.get(section, 'action_space'), None)
    #         num_agents = config.getint(section, 'num_agents', fallback=1)
    #         spec = SingleAgentRLModuleSpec(policy_class, obs_space, act_space) 
    #         configs_to_ray.update({'rl_module_spec': spec})

    policy_config.config_to_ray.update({'disable_env_checking': True})  # to avoid checking non-override default obs_space...
    policy_config.config_to_ray.update({"enable_worker_sync": False})

    # 5. assign termination conditions, terminate when achieving one of them
    stop_conditions = {}
    for k, v in config['STOP_CONFIG'].items():
        stop_conditions.update({k: int(v)})
        if k == "training_iteration":
            policy_config.config_to_ray.update({k: v})

    # 6. set up training function
    def trainer_fn(config):
        if policy_config.algo_ctor is not None:
            # If a dedicated config class exists (e.g., DreamerV3Config), use it
            print("\n\n\n\n\n\n\n\n\n\n policy_config.algo_cotr():", policy_config.algo_ctor())
            print("\n\n\n\n\n\n\n\n\n\n policy_config.algo_cotr().update_from_dict():", policy_config.algo_ctor().update_from_dict(config))
            print("\n\n\n\n\n\n\n\n\n\n policy_config.algo_ctor().update_from_dict().build():", policy_config.algo_ctor().update_from_dict(config).build())
            algo = policy_config.algo_ctor().update_from_dict(config).build()
        else:
            # Otherwise, fall back to the generic RLlib Algorithm.from_config
            algo = Algorithm.from_config(config)

        for i in range(config.get("training_iterations", 150)):
            result = algo.train()
            tune.report(**result)

    # 7. define resources
    resources = PlacementGroupFactory([
        {"CPU": 1.0},  # Learner actor
        {"CPU": 1.0},  # Env runner actor
    ])

    tune.run(
        config.get('ALG_CONFIG', 'alg_name'),
        config=policy_config.config_to_ray,
        # resources_per_trial=resources,
        checkpoint_freq=config.getint('RAY_CONFIG', 'checkpoint_freq'),  # number of iterations between checkpoints
        checkpoint_at_end=config.getboolean('RAY_CONFIG', 'checkpoint_at_end'),
        max_failures=config.getint('RAY_CONFIG', 'max_failures'),  # times to recover from the latest checkpoint
        stop=stop_conditions,
        local_dir="./ray_results/" + config.get('TRAIN_CONFIG', 'exp_name', fallback=env_name),
    )

    ray.shutdown()


if __name__ == '__main__':
    main(sys.argv[1:])
