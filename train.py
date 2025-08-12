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

def parse_args(args):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Parse argument used when running a DRL training for traffic control.",
        epilog="python train.py EXP_CONFIG")

    # ----required input parameters----
    parser.add_argument(
        '--exp-config', type=str, default='cotv_config.ini',
        help='Name of the experiment configuration file, as located in exp_configs.')

    # ----optional input parameters----
    parser.add_argument(
        '--log-level', type=str, default='ERROR',
        help='Level setting for logging to track running status.'
    )

    return parser.parse_known_args(args)[0]


def main(args):
    args = parse_args(args)
    logging.basicConfig(level=args.log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.info(f"DRL training with the following CLI args: {args}")

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
    
    # 3. set DRL algorithm/model
    # policy_config = PolicyConfig("CartPole-v1", config['ALG_CONFIG'], config['TRAIN_CONFIG'], config['MODEL_CONFIG'])
    policy_config = PolicyConfig(env_name, config['ALG_CONFIG'], config['TRAIN_CONFIG'], config['MODEL_CONFIG'])
    policy_config.config_to_ray.update({'disable_env_checking': True})  # to avoid checking non-override default obs_space...
    policy_config.config_to_ray.update({"enable_worker_sync": False})
    policy_config.algo_config.update_from_dict(policy_config.config_to_ray)


    tune.run(
        config.get('ALG_CONFIG', 'alg_name'),
        config=policy_config.algo_config,
        local_dir="./ray_results/" + config.get('TRAIN_CONFIG', 'exp_name', fallback=env_name),
    )

if __name__ == '__main__':
    main(sys.argv[1:])
