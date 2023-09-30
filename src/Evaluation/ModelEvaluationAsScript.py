#!/usr/bin/env python
# coding: utf-8

# # Evaluate Models
#
# Play each game with each model and save the results to a csv file. This file can be used to create plots and compare the models.

# In[45]:


import os
import sys
print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(['./../', './../../'])
import itertools
import random
import re
from builtins import range
from pathlib import Path

# Set the TensorFlow logging level to suppress debug messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm.auto import tqdm

import Training.TrainingScript as TrainingScript
from Ensemble import EnsembleMethods, Ensemble


# In[2]:


RESULTS_CSV = "./../../results.csv"
EVALUATION_RESULTS_CSV = "./../../evaluation_results.csv"
MODELS = "./../../models/"

NUM_REPETITIONS = 10
NUM_REPETITIONS_FOR_RANDOM_BASELINE = 250

N_FOR_TOP_N_ENSEMBLES = 3
M_FOR_SNAPSHOT_ENSEMBLES_AND_SOUPS = 3
ENSEMBLE_METHODS_USED = [
    EnsembleMethods.AVERAGE,
    EnsembleMethods.LOGISTIC_AVERAGE,
    # EnsembleMethods.AVERAGE_WITH_CONFIDENCE,
    # EnsembleMethods.LOGISTIC_AVERAGE_WITH_CONFIDENCE,
    EnsembleMethods.MAJORITY_VOTE,
]


# In[3]:


results_df = pd.read_csv(RESULTS_CSV)
results_df.head()


# In[4]:


results_df[results_df["training_model"] == "mnih2013"]


# In[5]:


list_of_games = list(results_df["game"].unique())
list_of_algorithms = list(results_df["training_model"].unique())

print(list_of_games, list_of_algorithms)


# In[6]:


evaluation_data_df = pd.DataFrame(columns=["game", "model", "model_id", "episode_rewards", "mean", "standard_deviation"])


# In[7]:


list_of_models = list(results_df["model_path"].unique())
models_dict = {
    model_path: tf.keras.models.load_model(MODELS + model_path, compile=False)
    for model_path in list_of_models
}


# ## Methods to Run Evaluations

# In[8]:


@tf.function(autograph=False)
def get_action_from_model(model, state):
    q_values = model(state)
    return tf.argmax(q_values, axis=1)


# In[9]:


@tf.function(autograph=False)
def get_action(model, state):
    state = tf.cast(tf.convert_to_tensor(state, dtype=tf.uint8), dtype=tf.float32) / 255.0
    state = tf.expand_dims(state, axis=0)
    q_values = model(state)
    return tf.argmax(q_values, axis=1)[0]


# In[10]:


def evaluate_model(game: str, model: tf.keras.Model, num_repetitions: int = 10):
    env = TrainingScript.create_env(game)
    # env = gym.wrappers.RecordVideo(env, video_folder='./video/', episode_trigger=lambda episode_id: episode_id % num_repetitions == 0)
    rewards = []
    for i in (tbar := tqdm(range(num_repetitions), leave=False)):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        step = 0
        while not done:
            if random.random() < 0.05:
                # To help the agent when it gets "stuck"
                # Happens mainly in Breakout caused by a bug in the game
                action = env.action_space.sample()
            else:
                step += 1
                # state = tf.convert_to_tensor(state, dtype=tf.float32) / 255.0
                # state = tf.expand_dims(state, axis=0)
                # action = get_action_from_model(model, state)
                # action = action.numpy()[0]
                action = get_action(model, state)

            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward

            tbar.set_description(f"Episode {i}  -  Step: {step}, Reward: {episode_reward}")

        rewards.append(episode_reward)
    return rewards


# In[11]:


def load_model(path: str):
    model = tf.keras.models.load_model(MODELS + path, compile=False)
    model.compile()
    return model


# In[12]:


def load_models(path_list: list[Path]) -> list:
    return [load_model(path) for path in path_list]


# In[13]:


def random_play(game: str, num_repetitions: int = 5):
    env = TrainingScript.create_env(game)
    rewards = []

    for _ in tqdm(range(num_repetitions), unit="episode", desc="Random play " + game.capitalize(), leave=False):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        step = 0
        while not done:
            step += 1

            state, reward, terminated, truncated, _ = env.step(env.action_space.sample())
            done = terminated or truncated
            episode_reward += reward

        rewards.append(episode_reward)

    return rewards


# ## Random Baseline for Each Game

# In[15]:


random_baseline = []
for game_name in tqdm(results_df["game"].unique(), unit="game", desc="Calculate random baseline"):
    rewards = random_play(game_name, num_repetitions=NUM_REPETITIONS_FOR_RANDOM_BASELINE)
    random_baseline.append(
        {
            "game": game_name,
            "model": "random play",
            "model_id": 0,
            "episode_rewards": rewards,
            "mean": np.mean(rewards),
            "standard_deviation": np.std(rewards)
        }
    )


# In[16]:


random_baseline_df = pd.DataFrame.from_records(random_baseline)
random_baseline_df


# ## Evaluate each Model on each Game

# In[17]:


evaluation_results = []
for game_name, model_name in (t := tqdm(list(itertools.product(list_of_games, list_of_algorithms)))):
    t.set_description(f"Evaluation models for {game_name.capitalize()} using {model_name.capitalize()}")
    fitting_models = list(
        results_df[(results_df["game"] == game_name) & (results_df["training_model"] == model_name)]["model_path"]
    )

    for idx, m_path in enumerate(fitting_models):
        model = models_dict[m_path]
        rewards = evaluate_model(game_name, model, num_repetitions=NUM_REPETITIONS)

        evaluation_results.append({
            "game": game_name,
            "model": model_name,
            "model_id": idx,
            "episode_rewards": rewards,
            "mean": np.mean(rewards),
            "standard_deviation": np.std(rewards),
            "model_path": m_path
        })


# In[18]:


evaluation_single_models_df = pd.DataFrame.from_records(evaluation_results)
evaluation_single_models_df.head()


# ## Evaluate as Ensembles

# In[19]:


models_used = set(list_of_algorithms) - {"random play", "interpretable_cnn"}


# In[20]:


def evaluate_ensemble(game: str, model: Ensemble, num_repetitions: int = 10):
    env = TrainingScript.create_env(game)
    rewards = []
    for i in (tbar := tqdm(range(num_repetitions), leave=False)):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        step = 0
        while not done:
            if random.random() < 0.05:
                # To help the agent when it gets "stuck"
                # Happens mainly in Breakout caused by a bug in the game
                action = env.action_space.sample()
            else:
                step += 1
                if model.ensemble_method == EnsembleMethods.MAJORITY_VOTE:
                    state = tf.convert_to_tensor(state, dtype=tf.float32) / 255.0
                    state = tf.expand_dims(state, axis=0)
                    q_values = model(state)
                    action = tf.argmax(q_values, axis=1)
                    # action = get_action_from_model(model, state)
                    action = action.numpy()[0]

                else:
                    action = get_action(model, state)

            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward

            tbar.set_description(f"Episode {i}  -  Step: {step}, Reward: {episode_reward}")

        rewards.append(episode_reward)
    return rewards


# ### Uniform Ensembles consisting of the best $n$ models of a flavor:

# In[21]:


evaluation_results_uniform_ensembles = []


# In[22]:


for game_name, model_name in (t := tqdm(list(itertools.product(list_of_games, models_used)),
                                        desc="Build and Ensemble Models")):
    t.set_description(f"Build and Ensemble Models ({model_name.capitalize()} for {game_name})")

    # Select Top-N models:
    model_selection = evaluation_single_models_df[
                          (evaluation_single_models_df["model"] == model_name) & (evaluation_single_models_df["game"] == game_name)
                      ].sort_values(by=["mean", "standard_deviation"], ascending=[False, True])[:N_FOR_TOP_N_ENSEMBLES]
    model_paths = [m_path for m_path in list(model_selection["model_path"])]
    models = [models_dict[m_path] for m_path in model_paths]

    for idx, ensemble_method in enumerate(ENSEMBLE_METHODS_USED):
        ensemble = Ensemble(models, ensemble_method)
        t.set_description(
            f"Build and Ensemble Models ({model_name.capitalize()} for {game_name} with {ensemble_method})")

        print(f"Build and Ensemble Models ({model_name.capitalize()} for {game_name} with {ensemble_method})")
        rewards = evaluate_ensemble(game_name, ensemble, NUM_REPETITIONS)
        evaluation_results_uniform_ensembles.append({
            "game": game_name,
            "model": f"Top-{N_FOR_TOP_N_ENSEMBLES} Ensemble ({ensemble_method}) with {model_name}",
            "model_id": idx,
            "episode_rewards": rewards,
            "mean": np.mean(rewards),
            "standard_deviation": np.std(rewards),
            "model_path": model_paths
        })


# In[23]:


evaluation_results_uniform_ensembles_df = pd.DataFrame.from_records(evaluation_results_uniform_ensembles)
evaluation_results_uniform_ensembles_df.head()


# ### Mixed Ensemble
#
# One Ensemble containing the best $n$ models of each model type (e.g. 3 models of Mnih 2013, 3 models of Mnih 2015, and 3 models of Mnih 2015 with Huber loss and Adam).

# In[24]:


evaluation_results_mixed_ensembles = []


# In[25]:


for game_name in (t := tqdm(list_of_games)):
    t.set_description(f"Evaluate mixed Ensemble for {game_name}")
    # Select Top-N models for each model type:
    model_paths = []
    for model_name in list_of_algorithms:
        model_selection = evaluation_single_models_df[
                          (evaluation_single_models_df["model"] == model_name) & (evaluation_single_models_df["game"] == game_name)
                        ].sort_values(by=["mean", "standard_deviation"], ascending=[False, True])[:N_FOR_TOP_N_ENSEMBLES]

        model_paths.extend([m_path for m_path in list(model_selection["model_path"])])

    models = [models_dict[m_path] for m_path in model_paths]
    for ensemble_method in ENSEMBLE_METHODS_USED:
        t.set_description(f"Evaluate mixed Ensemble ({ensemble_method}) for {game_name}")
        ensemble = Ensemble(models)
        rewards = evaluate_ensemble(game_name, ensemble, num_repetitions=NUM_REPETITIONS)
        evaluation_results_mixed_ensembles.append({
            "game": game_name,
            "model": f"Top-{N_FOR_TOP_N_ENSEMBLES} Mixed Ensemble ({ensemble_method})",
            "model_id": 0,
            "episode_rewards": rewards,
            "mean": np.mean(rewards),
            "standard_deviation": np.std(rewards),
            "model_path": model_paths
        })


# In[26]:


evaluation_results_mixed_ensembles_df = pd.DataFrame.from_records(evaluation_results_mixed_ensembles)
evaluation_results_mixed_ensembles_df.head()


# ### Snapshot Ensemble
#
# Ensembles consisting of the last $M$ training snapshots of a model. In the original paper, the snapshot get selected on the fly during training by saving models at local minima and increasing the learning rate after a model was selected. This leads to models that are more different from another and achieve higher results. This is not easily applicable for RL, so I use the $M$ newest snapshots instead.

# In[27]:


evaluation_results_snapshot_ensembles = []


# In[51]:


# def get_snapshots(dir) -> list:
#     if os.path.isfile(dir):
#         dir = os.path.dirname(dir)
#
#     print(dir)
#     snapshots = dict()
#     snapshot_re = re.compile(r"snapshot_(?P<idx>\d+)\.keras")
#
#     for file in Path(dir).glob("*.keras"):
#         f_name = str(file.name)
#         if snapshot_re.match(f_name):
#             snapshots[int(snapshot_re.match(f_name)["idx"])] = str(file)
#
#     snapshots[len(snapshots)] = str(Path(dir) / "snapshot_final.keras")
#     sorted_snapshots = list(zip(*sorted(list(snapshots.items()), key=lambda a: a[1])))[1]
#
#     print(sorted_snapshots)
#
#     return sorted_snapshots

def get_snapshots(models_dir) -> list:
    models_dir = models_dir[:-len("snapshot_final.keras")]

    snapshots = dict()
    snapshot_re = re.compile(r"snapshot_(?P<idx>\d+)\.keras")

    for file in Path(MODELS + models_dir).glob("*.keras"):
        f_name = str(file.name)
        if snapshot_re.match(f_name):
            snapshots[int(snapshot_re.match(f_name)["idx"])] = models_dir + f_name

    snapshots[len(snapshots)] = models_dir + "snapshot_final.keras"
    sorted_snapshots = list(zip(*sorted(list(snapshots.items()), key=lambda a: a[1])))[1]


    return sorted_snapshots


# In[ ]:


for game_name, model_name in (t := tqdm(list(itertools.product(list_of_games, models_used)),
                                        desc="Build Snapshot Ensemble")):
    t.set_description(f"Build Snapshot Ensemble ({model_name.capitalize()} for {game_name})")

    model_paths_for_game_and_model =         evaluation_single_models_df[(evaluation_single_models_df["model"] == model_name) & (evaluation_single_models_df["game"] == game_name)]["model_path"]

    for idx, unique_model_path in enumerate(model_paths_for_game_and_model.unique()):
        print(unique_model_path)
        model_paths = get_snapshots(unique_model_path)
        # Select M last snapshots:
        model_paths = model_paths[-M_FOR_SNAPSHOT_ENSEMBLES_AND_SOUPS:]
        assert len(model_paths) == M_FOR_SNAPSHOT_ENSEMBLES_AND_SOUPS
        models = load_models(model_paths)

        for idx, ensemble_method in enumerate(ENSEMBLE_METHODS_USED):
            ensemble = Ensemble(models, ensemble_method)

            rewards = evaluate_ensemble(game_name, ensemble, NUM_REPETITIONS)
            evaluation_results_snapshot_ensembles.append({
                "game": game_name,
                "model": f"{M_FOR_SNAPSHOT_ENSEMBLES_AND_SOUPS}-Snapshot Ensemble ({ensemble_method}) with {model_name}",
                "model_id": idx,
                "episode_rewards": rewards,
                "mean": np.mean(rewards),
                "standard_deviation": np.std(rewards),
                "model_path": model_paths
            })


# In[54]:


evaluation_results_snapshot_ensembles_df = pd.DataFrame.from_records(evaluation_results_snapshot_ensembles)
evaluation_results_snapshot_ensembles_df.head()


# ## Evaluate as Soups

# In[55]:


models_used = set(list_of_algorithms) - {"random play", "interpretable_cnn"}


# ### Uniform Soups created from the best $n$ models of a flavor:

# In[56]:


evaluation_results_uniform_soups = []


# In[57]:


from Soup import Soup

for game_name, model_name in (t := tqdm(list(itertools.product(list_of_games, models_used)))):
    t.set_description(f"Cooking the Soup Model ({model_name.capitalize()} for {game_name})")

    # Select Top-N models:
    model_selection = evaluation_single_models_df[
                          (evaluation_single_models_df["model"] == model_name) & (evaluation_single_models_df["game"] == game_name)
                        ].sort_values(by=["mean", "standard_deviation"], ascending=[False, True])[:N_FOR_TOP_N_ENSEMBLES]

    model_paths = [m_path for m_path in list(model_selection["model_path"])]
    models = [models_dict[m_path] for m_path in model_paths]

    soup = Soup(models).get_soup_model()

    rewards = evaluate_model(game_name, soup, NUM_REPETITIONS)
    evaluation_results_uniform_soups.append({
        "game": game_name,
        "model": f"Top-{N_FOR_TOP_N_ENSEMBLES} Soup of {model_name}",
        "model_id": 0,
        "episode_rewards": rewards,
        "mean": np.mean(rewards),
        "standard_deviation": np.std(rewards),
        "model_path": model_paths
    })


# In[58]:


evaluation_results_uniform_soups_df = pd.DataFrame.from_records(evaluation_results_uniform_soups)
evaluation_results_uniform_soups_df.head()


# ### Snapshot Soup
#
# Works like the Snapshot Ensemble, but as a Soup.

# In[59]:


evaluation_results_snapshot_soups = []


# In[60]:


for game_name, model_name in (t := tqdm(list(itertools.product(list_of_games, models_used)),
                                        desc="Cooking Soup Model")):
    t.set_description(f"Cooking Soup Model ({model_name.capitalize()} for {game_name})")

    model_paths_for_game_and_model =         results_df[(results_df["training_model"] == model_name) & (results_df["game"] == game_name)]["model_path"]

    for idx, unique_model_path in enumerate(model_paths_for_game_and_model.unique()):
        model_paths = get_snapshots(unique_model_path)
        # Select M last snapshots:
        model_paths = model_paths[-M_FOR_SNAPSHOT_ENSEMBLES_AND_SOUPS:]
        assert len(model_paths) == M_FOR_SNAPSHOT_ENSEMBLES_AND_SOUPS

        models = load_models(model_paths)

        soup = Soup(models).get_soup_model()

        rewards = evaluate_model(game_name, soup, NUM_REPETITIONS)
        evaluation_results_snapshot_soups.append({
            "game": game_name,
            "model": f"{M_FOR_SNAPSHOT_ENSEMBLES_AND_SOUPS}-Snapshot Soup {model_name}",
            "model_id": idx,
            "episode_rewards": rewards,
            "mean": np.mean(rewards),
            "standard_deviation": np.std(rewards),
            "model_path": model_paths
        })


# In[61]:


evaluation_results_snapshot_soups_df = pd.DataFrame.from_records(evaluation_results_snapshot_soups)
evaluation_results_snapshot_soups_df.head()


# ## Save the Evaluation Data

# In[62]:


evaluation_data_df = pd.concat([
    random_baseline_df,
    evaluation_single_models_df,
    # Ensembles:
    evaluation_results_uniform_ensembles_df,
    evaluation_results_mixed_ensembles_df,
    evaluation_results_snapshot_ensembles_df,
    # Soups:
    evaluation_results_uniform_soups_df,
    evaluation_results_snapshot_soups_df
])
evaluation_data_df.reset_index(drop=True, inplace=True)
evaluation_data_df.head()


# In[63]:


evaluation_data_df.to_csv(EVALUATION_RESULTS_CSV, index=False)

