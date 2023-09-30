# Playing Atari with Deep Reinforcement Learning: Comparing Individual Models with Ensembles and Soups

This repository accompanies my master's project at Ulm University. It contains the code for training and evaluating 
the models, as well as plotting the results of the experiments.

I used [Gymnasium](https://gymnasium.farama.org/index.html) for the RL environments and trained the 
models following the [2013 DQN paper](https://arxiv.org/abs/1312.5602) and 
[2015 follow-up paper](https://training.incf.org/sites/default/files/2023-05/Human-level%20control%20through%20deep%20reinforcement%20learning.pdf).
You can take a look into the written report [here](latex/DeepReinforcementLearning.pdf).

- Training the models: [`src/Training`](src/Training)
- Evaluating the models: [`src/Evaluation`](src/Evaluation)
- Plotting results and doing significance tests: [`src/Visualization`](src/Visualization)

Everything related to latex and all plots are in the [`latex/`](latex) folder. The Notebooks that were used to generate
the plots automatically save them to the latex folder.

All trained models and the results of the evaluation (>9GB) can be downloaded here: 
[Cloudstore Uni Ulm](https://cloudstore.uni-ulm.de/s/3fNHPR2kQowbdjz)
