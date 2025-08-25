from ray.rllib.models import ModelCatalog
from policies.models.CentralizedCriticModel import CentralizedCriticModel
import ast

from ray.tune.registry import register_trainable
from policies.drama.drama import Drama, DramaConfig
from ray.rllib.algorithms.dreamerv3 import DreamerV3Config
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.algorithm import Algorithm

def drama_config(train_configs, env_name, model_configs):
    return {
        # EMRAN hardcoded for now
        "env": env_name,
        "log_level": train_configs.get('log_level'),
        "batch_size_B": 4, # 16, 
        "batch_length_T": 4, # 64, 
        "gamma": 0.999,
        "model_size": "D",
        "training_ratio": 32, # 1024,
    }

def dreamerv3_config(train_configs, env_name, model_configs):
    return {
        # EMRAN hardcoded for now
        "env": env_name,
        "log_level": train_configs.get('log_level'),
        "batch_size_B": train_config.get('batch_size_B'), # 16, 
        "batch_length_T": train_config.get('batch_length_T'), # 64,
        "gamma": 0.999,
        "model_size": "XS",
        "training_ratio": train_config.get('training_ratio'), # 1024,
    }

def ppo_config(train_configs, env_name, model_configs):
    return {
        "env": env_name,
        "log_level": train_configs.get('log_level'),
        "num_workers": train_configs.getint('num_workers'),
        "train_batch_size": train_configs.getint('num_steps') * train_configs.getint('num_workers')
        if not train_configs.getint('train_batch_size') else train_configs.getint('train_batch_size'),
        "gamma": 0.999,   # discount rate
        "use_gae": True,
        "lambda": 0.97,
        "kl_target": 0.02,
        "num_sgd_iter": 10,
        "num_steps": train_configs.getint('num_steps'),
        "timesteps_per_iteration": train_configs.getint('num_steps') * train_configs.getint('num_workers'),
        "no_done_at_end": True,
    }


class PolicyConfig:
    def __init__(self, env_name, alg_configs, train_configs, model_configs=None):
        self.env_name = env_name
        self.name = alg_configs.get('alg_name')  # e.g., 'DreamerV3'
        self.config_to_ray = self.find_policy_config(train_configs, model_configs)
        self.algo_config = self.find_algo_config()

    def find_policy_config(self, train_configs, model_configs):
        if self.name == "Drama":
            return drama_config(train_configs, self.env_name, model_configs)
        elif self.name == 'DreamerV3':
            return dreamerv3_config(train_configs, self.env_name, model_configs)
        elif self.name == 'PPO':
            return ppo_config(train_configs, self.env_name, model_configs)
        else:
            raise ValueError(f"Unknown algorithm name: {self.name}")

    def find_algo_config(self):
        if self.name == "Drama":
            register_trainable("Drama", Drama)
            return DramaConfig()
        elif self.name == "DreamerV3":
            return DreamerV3Config()
        elif self.name == "PPO":
            return PPOConfig()
        else:
            return None  # fallback to Algorithm.from_config

