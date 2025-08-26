import os.path
import random
import numpy as np

import ray
from ray import tune
import argparse
import configparser
from collections import defaultdict, deque
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


# --- Driver-side evaluations ---
def eval_ppo_on_driver(algo, env_creator, episodes=10, explore=False):
    rets, lens = [], []
    for _ in range(episodes):
        env = env_creator({})
        out = env.reset()
        obs = out[0] if isinstance(out, tuple) else out  # gymnasium vs gym
        terminated = truncated = False
        ep_ret = 0.0
        ep_len = 0
        while not (terminated or truncated):
            action = algo.compute_single_action(obs, explore=explore)
            step = env.step(action)
            if len(step) == 5:
                obs, reward, terminated, truncated, _ = step
            else:
                obs, reward, done, _ = step
                terminated, truncated = done, False
            ep_ret += float(reward); ep_len += 1
        rets.append(ep_ret); lens.append(ep_len); env.close()
    return {
        "episodes": episodes,
        "episode_return_mean": float(np.mean(rets)),
        "episode_len_mean": float(np.mean(lens)),
    }


def eval_dreamerv3_with_module(algo, env_creator, episodes=5):
    is_discrete = hasattr(env_creator({}).action_space, "n")  # quick check
    """Run eval episodes using RLModule.forward_inference (no compute_single_action)."""
    lw = algo.workers.local_worker()
    mod = getattr(lw, "module", None)
    if mod is None:
        # Fallback for odd setups; some builds expose a mapping
        modules = getattr(lw, "modules", None) or getattr(lw, "module_dict", None)
        if isinstance(modules, dict) and len(modules) == 1:
            mod = next(iter(modules.values()))
        else:
            raise RuntimeError("Could not locate RLModule on local worker.")
    results = {"episodes": 0, "episode_return_mean": 0.0, "episode_len_mean": 0.0}
    returns, lengths = [], []

    for _ in range(episodes):
        env = env_creator({})
        out = env.reset()
        obs = out[0] if isinstance(out, tuple) else out
        # Initial recurrent state for batch_size=1
        state = mod.get_initial_state() # EMRAN changed this
        ep_ret, ep_len = 0.0, 0
        terminated = truncated = False
        is_first = np.ones((1,))
        while not (terminated or truncated):
            # Build a batch dict for forward_inference
            batch = {
                Columns.STATE_IN: tree.map_structure(
                        lambda s: tf.convert_to_tensor(s), state
                    ),
                Columns.OBS: tf.convert_to_tensor(np.expand_dims(obs, 0)),
                "is_first": tf.convert_to_tensor(is_first),
            }

            # forward pass (inference-time)
            out = mod.forward_inference(batch)
            # out is a dict; DreamerV3 returns actions as one-hot (discrete) or scaled tensor (continuous)
            act_tensor = out["actions"]  # shape (B, ...) with B=1
            if is_discrete:
                # DreamerV3 returns one-hot => convert to index
                action = int(np.argmax(np.array(act_tensor)[0]))
            else:
                # Continuous: typically in [-1, 1]; may need to rescale to env.action_space
                action = np.array(act_tensor)[0]
                # Optional rescale example:
                # low, high = env.action_space.low, env.action_space.high
                # action = low + (0.5 * (action + 1.0) * (high - low))

            step = env.step(action)
            if len(step) == 5:  # gymnasium
                obs, reward, terminated, truncated, _info = step
            else:               # old gym
                obs, reward, done, _info = step
                terminated, truncated = done, False
            ep_ret += float(reward); ep_len += 1
            is_first = np.zeros((1,))

        returns.append(ep_ret); lengths.append(ep_len)
        env.close()

    results["episodes"] = episodes
    results["episode_return_mean"] = float(np.mean(returns))
    results["episode_len_mean"] = float(np.mean(lengths))
    return results

def eval_drama_with_module(algo, env_creator, episodes=5):
    is_discrete = hasattr(env_creator({}).action_space, "n")  # quick check
    """Run eval episodes using RLModule.forward_inference (no compute_single_action)."""
    lw = algo.workers.local_worker()
    mod = getattr(lw, "module", None)
    if mod is None:
        # Fallback for odd setups; some builds expose a mapping
        modules = getattr(lw, "modules", None) or getattr(lw, "module_dict", None)
        if isinstance(modules, dict) and len(modules) == 1:
            mod = next(iter(modules.values()))
        else:
            raise RuntimeError("Could not locate RLModule on local worker.")
    results = {"episodes": 0, "episode_return_mean": 0.0, "episode_len_mean": 0.0}
    returns, lengths = [], []

    for _ in range(episodes):
        env = env_creator({})
        out = env.reset()
        obs = out[0] if isinstance(out, tuple) else out
        # Initial recurrent state for batch_size=1
        current_obs = obs
        context_obs = deque(maxlen=16) # EMRAN hardcoded length for now 
        context_obs.append(np.expand_dims(obs, 0))
        context_action = deque(maxlen=16) # EMRAN hardcoded length for now
        context_action.append(np.expand_dims(env.action_space.sample(), 0)) 
        state = {
            "context_obs": tf.stack(list(context_obs), axis=0),
            "context_action": tf.stack(list(context_action), axis=0)
        }
        ep_ret, ep_len = 0.0, 0
        terminated = truncated = False
        is_first = np.ones((1,))
        while not (terminated or truncated):
            # Build a batch dict for forward_inference
            batch = {
                    Columns.STATE_IN: tree.map_structure(
                        lambda s: tf.convert_to_tensor(s), state
                    ),
                    Columns.OBS: tf.convert_to_tensor(np.expand_dims(obs, 0)),
                    "is_first": tf.convert_to_tensor(is_first),
                }

            # forward pass (inference-time)
            out = mod.forward_inference(batch)
            # out is a dict; DreamerV3 returns actions as one-hot (discrete) or scaled tensor (continuous)
            act_tensor = out["actions"]  # shape (B, ...) with B=1
            if is_discrete:
                # DreamerV3 returns one-hot => convert to index
                action = int(np.argmax(np.array(act_tensor)[0]))
            else:
                # Continuous: typically in [-1, 1]; may need to rescale to env.action_space
                action = np.array(act_tensor)[0]
                # Optional rescale example:
                # low, high = env.action_space.low, env.action_space.high
                # action = low + (0.5 * (action + 1.0) * (high - low))

            step = env.step(action)
            if len(step) == 5:  # gymnasium
                obs, reward, terminated, truncated, _info = step
            else:               # old gym
                obs, reward, done, _info = step
                terminated, truncated = done, False
            context_obs.append(np.expand_dims(obs, 0))
            context_action.append(np.expand_dims(action, 0))
            state = {
                "context_obs": tf.stack(list(context_obs), axis=0),
                "context_action": tf.stack(list(context_action), axis=0)
            }
            ep_ret += float(reward); ep_len += 1
            is_first = np.zeros((1,))

        returns.append(ep_ret); lengths.append(ep_len)
        env.close()

    results["episodes"] = episodes
    results["episode_return_mean"] = float(np.mean(returns))
    results["episode_len_mean"] = float(np.mean(lengths))
    return results