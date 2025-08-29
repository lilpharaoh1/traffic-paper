# Drama Traffic Project

## Setup
We have used a conda environment for this work. To set up the conda environment we have used, please run the following command. 

```python
conda env create -f environment.yml
```

To then enter this environment, use the command

```python
conda activate drama-traffic
```

## Training
Experiments are configured using the files in `exp_configs/`. To run an experiment with a specific model, use the command

```python
python train.py --exp-config [ppo/dreamerv3/drama]_config.ini

```

To run with the `sumo-gui` activated, edit the `sumo-gui` field in the config file to True and run using the appropriate command above.

To continue training from a checkpoint, please use the `--restore-path` flag followed by the path to the checkpoint. 

```python
python train.py --exp-config [ppo/dreamerv3/drama]_config.ini --restore-path /path/to/checkpoint_XXX

```

If GPU use is required, please run the training script with the `--use-gpu` flag. The number of GPUs and GPU workers to use is specified in the training script.

## Evaluation
Evaluation is run using a training checkpoint. The `--restore-path` flag must be used to point towards the checkpoint to be evaluated. The `--exp-config` flag must also be used. The configuration file should specify the number of episodes the agent should be evaluated using, set with the `eval_duration` parameter. At the end of script, the evaluation results are printed to the terminal.

The evaluation script may be ran using the following command. The `--use-gpu` flag may also be enabled for GPU use.

```python
python eval.py --exp-config [ppo/dreamerv3/drama]_config.ini --restore-path /path/to/checkpoint_XXX

``` 

Due to compatibility issues between DreamerV3 and the legacy method by which other agents are evaluated in Ray, the agents are instead evaluated using the RLModule `forward_inference` method (DreamerV3 and Drama) or by using `compute_single_action` method (PPO) and manually stepping the environment forward. The evaluation functions can be found in the file `eval_fns.py`.


## Notes
### Iterations
In Ray, an iteration (`iter`) refers to every time `algo.train()` is called. This may happen at different frequencies for different algorithms. For example, after an initial period to fill up it's replay buffer, both MBRL algorithms call `algo.train()` every timestep. This is in comparison to PPO which calls `algo.train()` every episode. It is for this reason that the length of training should likely be specified using `train_timesteps` rather then `train_iteration`. This is done in the configuration file.

### Environment
I essentially reeled back the CoTV environment to 1) make the environment single agent and therefore compatible with the DreamerV3 implementation available in Rat and to 2) use just the traffic light information i.e. removing the information shared by the CAV.

#### Observations
The agent receives as its observation an 1D vector containing the current phase of the traffic light concatenated with the number of vehicles on each incoming and outgoing road, this information is repeated for each set of lights in the network. The resulting observation is a long vector containing the overall traffic state at each set of lights. Yes, this is not an efficient way of representing traffic state, however was just a starting point and should be changed in future. 

The RL agent implementations, including the MBRL agents, use a MLP-based encoder for the observations so the vector representation currently is compatible.   

##### Actions
The agent outputs a vector in which each element represents whether a corresponding set of traffic lights in the network should be advanced to the next phase. Therefore, if there are N sets of traffic light the action vector will be of length N.

The DreamerV3 implementation in Ray 2.10.0, and therefore the Drama implementation too, is only compatible with the action spaces gymnasium.spaces.Box and gymnasium.spaces.Discrete. To work around this, I have implemented the action space using a Box type (continuous multi-dimensional). The range of the action space is limited to [0.0, 1.0], and before the action is passed to sumo I map values [0.0, 0.5) to a "don't change the phase" action and values [0.5, 1.0] to a "change the phase" action. Ideally, the output should be of type gymnasium.spaces.MultiDiscrete, however this is not implemented yet. 

##### Rewards
The reward is implemented in the same manner as the traffic light reward described in the original CoTV paper. The total reward is not divided by the road capacity as described in the paper, however I could not find in the CoTV implementation where this was done either. So if it was done in the CoTV implementation, it very well could be done unknown to myself, in this implementation. Perhaps something to investigate when adapting the reward function. 

### CoTV Evaluation
I have left the `evaluation/` directory from the original CoTV repo. I haven't touched or used it at all, but there seems to be some tools for extracting some SUMO related metrics like CO2 emissions, delay, trip length, etc. These tool could likely be adapted to work off of some output of the `eval.py` file. I haven't looked into this but would surely be useful for a paper especially if the paper motivation is demonstrating a use case for Drama.
 
### Training with Config Files
The training and evaluation is controlled primarily using config files, found in `exp_configs/`. A description of the parameters can be found in the files. Performing training with one of these files will create a directory containing the training and evaluation results at `~/ray_results/AGENTNAME_ENVNAME_DATETIME`. 

The agent weights are checkpointed at a frequency specified in the config file. If training is stopped/completed and you wish to continue training further, use the `--restore-path` shown above.  

### MBRL Algorithms
Both the DreamerV3 and Drama models are largely set up in the same way, bar what they use for world modeling (`tf/models/world_model.py::WorldModel::sequence_model`). For explaination purposes we will refer only to the Drama implementation when discussing how the agent is set up.

...Write more once everything is cleaned up...  

### DreamerV3Traffic vs DreamerV3
In this version of Ray (2.10.0), DreamerV3 uses a different training and evaluation procedure than the other agents (e.x. PPO). Therefore if one were to a standard Ray training and evaluation script it would crash for DreamerV3. Some mostly inconsequential methods and variables are not present in the DreamerV3 agent that are required to run these scripts. To quickly and easily fix this, I simply added these methods to a copy of the DreamerV3 agent and registered it as a custom algorithm. This agent is called DreamerV3Traffic. Currently (29/08/2025), no other changes are present between the two methods.

In future, if one wanted to make a multi-agent implementation of this work rather than a single-agent implementation, I would change this DreamerV3Traffic agent or make a copy. Will save you some time messing about with registering a custom algo.

   

## TODO
- [ ] Clean up README
- [ ] Clean up comments
- [X] Build evaluation script
- [X] Validate model saving/loading
- [X] Check MBRL configs loaded correctly
- [X] Fix Drama reward reporting
- [X] Implement DreamerV3 as custom algorithm
- [ ] Implement MultiDiscrete for DreamerV3 + Drama
- [ ] Implement multiagent DreamerV3 + Drama
- [ ] Align reward graphs on wandb (iterations)
- [ ] Investigate sumo variables (vehicle load, etc.)
- [ ] Test with Dublin layout
- [ ] Build larger Dublin layouts 
- [ ] Retrain PPO with new reward
- [ ] Retrain DreamerV3 with new reward
- [ ] Retrain Drama with new reward

## Contact
email: moustafe@tcd.ie / e.y.moustafa@ed.ac.uk

linkedin: Emran Yasser Moustafa
