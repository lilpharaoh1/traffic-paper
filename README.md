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

## Running
Experiments are configured using the files in `exp_configs/`. To run an experiment with a specific model, use the command

```python
python train.py --exp-config [ppo/dreamerv3/drama]_config.ini

```

To run with the `sumo-gui` activated, edit the `sumo-gui` field in the config file to True and run using the appropriate command above.

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
