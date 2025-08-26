import os.path
import random
import numpy as np

import ray
from ray import tune
import argparse
import configparser
import sys
import logging
import tree
from ray.tune.registry import register_env
from ray.tune import PlacementGroupFactory
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.rllib.algorithms.registry import get_algorithm_class

from scenario.scen_retrieve import SumoScenario
from envs.env_register import register_env_gym
from policies import PolicyConfig
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.core.columns import Columns
from ray.rllib.utils.framework import try_import_tf
_, tf, _ = try_import_tf()

from eval_fns import *


def parse_args(args):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Parse arguments for evaluating an RLlib checkpoint on the SUMO env.",
        epilog="python eval.py --exp-config EXP_CONFIG --restore-path /abs/path/to/CHECKPOINT"
    )

    # ----required input parameters----
    parser.add_argument(
        '--exp-config', type=str, default='dev_config.ini',
        help='Name of the experiment configuration file, as located in exp_configs.'
    )

    # ----optional input parameters----
    parser.add_argument(
        '--restore-path', type=str, default=None,
        help='Absolute path to checkpoint folder (checkpoint_XXX/) from ~/, located in ray_results.'
    )

    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Use this flag to enable GPU usage."
    )

    parser.add_argument(
        '--log-level', type=str, default='ERROR',
        help='Level setting for logging to track running status.'
    )

    return parser.parse_known_args(args)[0]


def main(args):
    args = parse_args(args)

    logging.basicConfig(level=args.log_level,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.info(f"DRL evaluation with the following CLI args: {args}")

    if not args.restore_path:
        raise ValueError("--restore-path is required for evaluation.")

    if not os.path.isdir(args.restore_path):
        raise FileNotFoundError(f"Checkpoint directory not found: {args.restore_path}")

    ray.init()

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
    policy_config.config_to_ray.update({'disable_env_checking': True})
    policy_config.config_to_ray.update({"enable_worker_sync": False})
    policy_config.config_to_ray.update({"enable_lazy_rl_module_building": False})
    policy_config.algo_config.update_from_dict(policy_config.config_to_ray)

    # 4. set training and evalation details
    eval_dict = {
        "evaluation_interval": config.getint('TRAIN_CONFIG', 'eval_interval'), 
        "evaluation_duration": config.getint('TRAIN_CONFIG', 'eval_duration'),
        "evaluation_duration_unit": "episodes",
        "evaluation_config": {
            "explore": False
        },
        "evaluation_num_workers": 0, # EMRAN problems with build otherwise for some reason
        "enable_async_evaluation": True,
        "create_env_on_local_worker": True,
        "create_env_on_driver": True,
    }
    policy_config.algo_config.update_from_dict(eval_dict)

    # 5. set GPU support
    if args.use_gpu:
        policy_config.algo_config.resources(num_gpus=1)
        policy_config.algo_config.resources(num_learner_workers=1, num_gpus_per_learner_worker=1)

    # ---- Build, restore, evaluate ----
    # Build the Algorithm from the AlgorithmConfig
    algo = policy_config.algo_config.build()

    # Restore from the provided checkpoint directory
    logger.info(f"Restoring from checkpoint: {args.restore_path}")
    algo.restore(args.restore_path)

    def find_eval_fn(algo_name, algo, env_creator):
        episodes = config.getint('TRAIN_CONFIG', 'eval_duration')
        if algo_name == "Drama":
            return eval_drama_with_module(algo, env_creator, episodes=episodes)
        elif algo_name == "DreamerV3":
            return eval_dreamerv3_with_module(algo, env_creator, episodes=episodes)
        elif algo_name == "PPO":
            return eval_ppo_on_driver(algo, env_creator, episodes=episodes)
        else:
            return None 

    metrics = find_eval_fn(
        policy_config.name, algo, this_env_register
    )
    print("Evaluation Metrics:", metrics)

    # Optional: shut down Ray
    ray.shutdown()

if __name__ == '__main__':
    main(sys.argv[1:])

    # EMRAN below would work for PPO but not MBRL. This is due to some
    # EMRAN compatiility issues with DreamerV3 using a custom RLModule
    # # ---- Build, restore, evaluate ----
    # # Build the Algorithm from the AlgorithmConfig
    # algo = policy_config.algo_config.build()

    # # Restore from the provided checkpoint directory
    # logger.info(f"Restoring from checkpoint: {args.restore_path}")
    # algo.restore(args.restore_path)

    # # Run evaluation using the evaluation_* settings we applied above
    # logger.info("Running evaluation...")
    # eval_results = algo.evaluate()
    # # Pretty print key metrics to stdout
    # # (Available keys typically include "evaluation" dict with "episode_reward_mean", etc.)
    # print("\n=== Evaluation Results ===")
    # try:
    #     # Newer RLlib returns nested dicts like {"evaluation": {"episode_reward_mean": ...}}
    #     print(json.dumps(eval_results, indent=2))
    # except TypeError:
    #     # Fallback printing if non-serializable
    #     print(eval_results)

    # # Save results next to the checkpoint for convenience
    # out_path = os.path.join(args.restore_path, "eval_results.json")
    # try:
    #     with open(out_path, "w") as f:
    #         json.dump(eval_results, f, indent=2)
    #     print(f"\nSaved evaluation results to: {out_path}")
    # except TypeError:
    #     # If there are non-serializable objects, save a simplified version
    #     simple = {k: v for k, v in eval_results.items() if isinstance(v, (int, float, str, dict, list))}
    #     with open(out_path, "w") as f:
    #         json.dump(simple, f, indent=2)
    #     print(f"\nSaved simplified evaluation results to: {out_path}")
