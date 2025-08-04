from pathlib import Path

from envs.env_cotv import CoTVEnv

import ray
from ray.tune.registry import register_env
from ray.rllib.algorithms import dreamerv3

import os.path
import random

import argparse
import configparser
import sys
import logging
from ray.tune.registry import register_env
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.rllib.algorithms.registry import get_algorithm_class

from scenario.scen_retrieve import SumoScenario
from envs.env_register import register_env_gym
from policies import PolicyConfig

def env_creator(env_config):
    return CoTVEnv(config)  # return an env instance



if __name__ == "__main__":
    # env = TradingEnv(config)
    ray.init()

    # import experiment configuration
    config_file = 'exp_configs/cotv_config.ini'
    config = configparser.ConfigParser()
    config.read(os.path.join('./exp_configs', config_file))

    # 1. process SUMO scenario files to get info, required in make env
    scenario = SumoScenario(config['SCEN_CONFIG'])
    logger.info(f"The scenario cfg file is {scenario.cfg_file_path}.")

    # 2. register env and make env in OpenAIgym, then register_env in Ray
    this_env = __import__("envs", fromlist=[config.get('TRAIN_CONFIG', 'env')])
    if hasattr(this_env, config.get('TRAIN_CONFIG', 'env')):
        this_env = getattr(this_env, config.get('TRAIN_CONFIG', 'env'))
    register_env("my_env", env_creator)
    trainer = dreamerv3.DreamerV3Trainer(env="my_env")

    while True:
        print(trainer.train())
