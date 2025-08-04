# train_dreamer.py

from ray.tune.registry import register_env
from ray import tune, init
from ray.rllib.algorithms.dreamerv3 import DreamerV3Config
import gym
from gym.spaces import Discrete, Box
import numpy as np

# Define the custom environment
class MyCustomEnv(gym.Env):
    def __init__(self, config):
        print("[DEBUG] MyCustomEnv.__init__ called")
        self.observation_space = Box(low=0, high=1, shape=(4,), dtype=np.float32)
        self.action_space = Discrete(2)

    def reset(self):
        print("[DEBUG] reset called")
        return np.zeros(4, dtype=np.float32)

    def step(self, action):
        print(f"[DEBUG] step called with action={action}")
        return np.zeros(4, dtype=np.float32), 0.0, True, {}

# Register the environment BEFORE training
def env_creator(env_config):
    print("[DEBUG] env_creator called")
    return MyCustomEnv(env_config)

if __name__ == "__main__":
    init()

    register_env("my_custom_env", env_creator)

    config = (
        DreamerV3Config()
        .environment(env="my_custom_env", env_config={})
        # .framework("torch")  # DreamerV3 only supports torch
        .training(model_size="S", batch_size_B=16)  # "S" = small model, adjust as needed
    )

    algo = config.build()
    result = algo.train()
    print(result)

