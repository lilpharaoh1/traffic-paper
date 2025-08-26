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

Due to compatibility issues between DreamerV3 and the legacy method by which other agents are evaluated in Ray, the agents are instead evaluated using the RLModule `forward_inference` method (DreamerV3 and Drama) or by using `compute\_single\_action` method (PPO) and manually stepping the environment forward. The evaluation functions can be found in the file `eval_fns.py`.
  

## TODO
- [ ] Clean up README
- [ ] Clean up comments
- [X] Build evaluation script
- [X] Validate model saving/loading
- [X] Check MBRL configs loaded correctly
- [X] Fix Drama reward reporting
- [ ] Align reward graphs on wandb (iterations)
- [ ] Implement sumo vehicle load, etc.
- [ ] Test with Dublin layout
- [ ] Build larger Dublin layouts 
- [ ] Retrain PPO with new reward
- [ ] Retrain DreamerV3 with new reward
- [ ] Retrain Drama with new reward

## Contact
email: moustafe@tcd.ie / e.y.moustafa@ed.ac.uk

linkedin: Emran Yasser Moustafa
