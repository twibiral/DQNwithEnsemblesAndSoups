import argparse
import os
import random
import time
from datetime import datetime
from pathlib import Path

# Set the TensorFlow logging level to suppress debug messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras import backend as K

K.set_floatx('float32')

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from Training import CSV_Database, ModelFactory
from Training.DQN_Agent import DQNAgent
from Training.ReplayBuffer import ReplayBuffer, PixelNormalizationBufferWrapper

# Define the constants with default values
EPSILON_DECAY_SPAN = 0.1
EPS_START = 1.0
EPS_END = 0.01
NUM_FRAMES = 10_000_000
BATCH_SIZE = 32
GAMMA = 0.99
BUFFER_SIZE = 1_000_000
MODEL_CHECKPOINT_DIR = "../../models/"
RESULTS_TABLE = "./../../results.csv"
DQN_TARGET_UPDATE_FREQUENCY = 10_000  # How often to update the target network in frames
START_LEARNING_AFTER = 0  # How many frames to wait before starting to learn
SNAPSHOT_FREQUENCY = 100_000  # How often to save a snapshot of the model in frames (Snapshot every n frames)
VERBOSE = False

available_atari_games = {
    "breakout": "BreakoutNoFrameskip-v4",
    "pong": "PongNoFrameskip-v4",
    "enduro": "EnduroNoFrameskip-v4",
    "seaquest": "SeaquestNoFrameskip-v4",
    "space_invaders": "SpaceInvadersNoFrameskip-v4",
    # Problematic because of laser blinking (needs 3 frames instead of 4)
    "asteroids": "AsteroidsNoFrameskip-v4",
}

# mnih2013 is the normal DQN; mnih2015 and with_huber_loss_and_adam use double DQN
available_training_versions = {
    "mnih2013": ModelFactory.get_model_mnih2013,
    "mnih2015": ModelFactory.get_model_mnih2015,
    "with_huber_loss_and_adam": ModelFactory.dqn_with_huber_loss_and_adam,
    "interpretable_cnn": ModelFactory.get_interpretable_cnn,
}
modes_with_double_dqn = ["mnih2015", "with_huber_loss_and_adam", "interpretable_cnn"]

ATARI_GAME = list(available_atari_games.keys())[0]  # => breakout / BreakoutNoFrameskip-v4
TRAINING_MODE = list(available_training_versions.keys())[0]  # => mnih2013


def train(agent: DQNAgent, env, num_frames: int, batch_size: int, decay_span: int, snapshot_dir: str,
          snapshot_freq: int, target_network_update_freq=DQN_TARGET_UPDATE_FREQUENCY,
          start_learning_after=START_LEARNING_AFTER, verbose=VERBOSE):
    """
    Train the agent on the environment for the specified number of episodes
    """
    assert not Path(snapshot_dir).is_file()

    if not Path(snapshot_dir).exists():
        Path(snapshot_dir).mkdir(parents=True, exist_ok=True)
    snapshot_base = os.path.join(snapshot_dir, "snapshot_%d.keras")

    # linear decay of epsilon (Subtracted after each learning step)
    eps_linear_decay = (EPS_START - EPS_END) / (num_frames * decay_span)
    epsilon = EPS_START
    current_state, _ = env.reset()
    episode_reward = 0
    episode_rewards = []
    loss_history = []
    snapshot_counter = 0

    for time_step in range(num_frames):
        # ACT
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = agent.act(current_state, scaling_factor=1/255.0)

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        episode_reward += reward

        # STORE in memory
        agent.memory.add(current_state, reward, next_state, done)

        # UPDATE state (reset if done)
        if done:
            current_state, _ = env.reset()
            episode_rewards.append(episode_reward)
            episode_reward = 0
            if not verbose:
                print(f"Episode {len(episode_rewards):>3} finished. Reward: {episode_rewards[-1]:>7.1f} "
                      f"(Best: {np.max(episode_rewards):>5}, ε: {epsilon:.2f})")

        else:
            current_state = next_state

        # TRAIN agent if buffer is full
        if agent.memory.is_filled() and time_step >= start_learning_after:
            loss = agent.optimise_td_loss(batch_size)
            loss_history.append(loss)
            epsilon = max(epsilon - eps_linear_decay, EPS_END)

        # Update the target network only if Double DQN is used
        if agent.double_dqn and agent.memory.is_filled() and time_step >= start_learning_after \
                and time_step % target_network_update_freq == 0:
            agent.update_target_network()

        # Save a snapshot of the model
        if time_step % snapshot_freq == 0:
            agent.q_network.save(snapshot_base % snapshot_counter)
            snapshot_counter += 1

        # Print info
        if verbose:
            additional_info = f"  |  reward: {episode_reward:>7.1f} - best: {np.max(episode_rewards):>5}" \
                if len(episode_rewards) > 0 else ""
            print(f"Frame {time_step + 1:>5} -- Episode {len(episode_rewards):>3}" + additional_info +
                  f"  |  ε: {epsilon:.2f}", end="\r")

    return episode_rewards, loss_history


def main(decay_span, num_frames, batch_size, gamma, buffer_size, model_checkpoint_dir, snapshot_freq, results_table,
         atari_game, training_mode, dqn_target_update_frequency, start_learning_after, verbose=VERBOSE):
    start_execution_time = time.time()
    env = create_env(atari_game)
    print("Created", atari_game, "environment for", training_mode)

    use_double_dqn = training_mode in modes_with_double_dqn

    # Set model checkpoint dir
    start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_checkpoint_dir = os.path.join(model_checkpoint_dir, f"{training_mode}/{atari_game}/started_at_{start_time}")

    # Create the agent
    model = available_training_versions[training_mode](env)
    buffer = PixelNormalizationBufferWrapper(ReplayBuffer(buffer_size))
    agent = DQNAgent(buffer, model, gamma, double_dqn=use_double_dqn)

    # Train the agent
    episode_rewards, loss_history = train(agent, env, num_frames, batch_size, decay_span,
                                          snapshot_freq=snapshot_freq, snapshot_dir=model_checkpoint_dir,
                                          target_network_update_freq=dqn_target_update_frequency,
                                          start_learning_after=start_learning_after,
                                          verbose=verbose)

    # Save the model
    final_model_path = os.path.join(model_checkpoint_dir, "snapshot_final.keras")
    agent.q_network.save(final_model_path)

    # Save Training results
    CSV_Database.append_to_csv(csv_path=results_table, input_row=[
        atari_game, training_mode, episode_rewards, loss_history, final_model_path
    ])

    save_plot(atari_game, episode_rewards, loss_history, model_checkpoint_dir, training_mode)

    elapsed_time = time.time() - start_execution_time
    print('Training time:', time.strftime("%Hh:%Mm:%Ss", time.gmtime(elapsed_time)))


def create_env(atari_game):
    # Create the environment
    env = gym.make(available_atari_games[atari_game], full_action_space=False)
    env = gym.wrappers.AtariPreprocessing(env, noop_max=30, frame_skip=4, screen_size=84, terminal_on_life_loss=False,
                                          grayscale_obs=True, grayscale_newaxis=False,
                                          scale_obs=False)  # Frame stacking
    env = gym.wrappers.FrameStack(env, 4)
    return env


def save_plot(atari_game, episode_rewards, loss_history, model_checkpoint_dir, training_mode):
    _, ax1 = plt.subplots()
    sns.lineplot(data=episode_rewards, label="episode reward", ax=ax1)
    sns.regplot(x=list(range(len(episode_rewards))), y=episode_rewards, x_ci="cd", scatter=False,
                label="regression line", ax=ax1)
    plt.title(f"Rewards per Episode for {training_mode} on {atari_game}")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.savefig(os.path.join(model_checkpoint_dir, "rewards_per_episode.pdf"))
    plt.close()

    _, ax1 = plt.subplots()
    sns.lineplot(data=loss_history, label="loss per Training step", ax=ax1)
    sns.regplot(x=list(range(len(loss_history))), y=loss_history, x_ci="cd", scatter=False,
                label="regression line", ax=ax1)
    plt.title(f"Loss per Training Step for {training_mode} on {atari_game}")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(model_checkpoint_dir, "loss_per_episode.pdf"))
    plt.close()


if __name__ == "__main__":
    # Parse all arguments and pass them to main(...) where the actual Training happens
    parser = argparse.ArgumentParser(description="Train a DQN agent on an Atari game.")

    parser.add_argument("-e", "--epsilon_decay_span", type=float, default=EPSILON_DECAY_SPAN,
                        help=f"Set the value of EPSILON_DECAY_SPAN. The decay span defines how fast the exploration "
                             f"probability declines. After EPSILON_DECAY_SPAN * NUM_FRAMES frames the epsilon is "
                             f"fixed to 0.1 (default: {EPSILON_DECAY_SPAN})")
    parser.add_argument("-n", "--num_frames", type=int, default=NUM_FRAMES,
                        help=f"Set the value of NUM_FRAMES (default: {NUM_FRAMES})")
    parser.add_argument("-b", "--batch_size", type=int, default=BATCH_SIZE,
                        help=f"Set the value of BATCH_SIZE (default: {BATCH_SIZE})")
    parser.add_argument("-g", "--gamma", type=float, default=GAMMA, help=f"Set the value of GAMMA (default: {GAMMA})")
    parser.add_argument("-u", "--buffer_size", type=int, default=BUFFER_SIZE,
                        help=f"Set the value of BUFFER_SIZE (default: {BUFFER_SIZE})")
    parser.add_argument("-m", "--model_checkpoint_dir", type=str, default=MODEL_CHECKPOINT_DIR,
                        help=f"Set where model checkpoints are stored (default: {MODEL_CHECKPOINT_DIR})")
    parser.add_argument("-r", "--results_table", type=str, default=RESULTS_TABLE,
                        help=f"Set the path to the csv where the results are stored. The csv is created if it doesn't "
                             f"exist yet (default: {RESULTS_TABLE})")
    parser.add_argument("-a", "--atari_game", type=str, default=ATARI_GAME, choices=available_atari_games.keys(),
                        help=f"Chose the Atari game to play (default: {ATARI_GAME}).")
    parser.add_argument("-t", "--training_mode", type=str, default=TRAINING_MODE,
                        choices=available_training_versions.keys(),
                        help=f"Chose the model and Training type (default: {TRAINING_MODE}).")
    parser.add_argument("-f", "--dqn_target_update_frequency", type=int, default=DQN_TARGET_UPDATE_FREQUENCY,
                        help=f"Set the frequency of updating the target network in frames "
                             f"(default: Update after {DQN_TARGET_UPDATE_FREQUENCY} frames)")
    parser.add_argument("-s", "--start_learning_after", type=int, default=START_LEARNING_AFTER,
                        help=f"Set the number of frames to wait before starting to learn "
                             f"(default: {START_LEARNING_AFTER})")
    parser.add_argument("-sf", "--snapshot_frequency", type=int, default=SNAPSHOT_FREQUENCY,
                        help=f"Set the frequency of saving a snapshot of the model in frames. Save a snapshot "
                             f"every n frames (default: {SNAPSHOT_FREQUENCY}).")
    parser.add_argument("-v", "--verbose", action="store_true", default=VERBOSE,
                        help="Print more information during Training.")
    args = parser.parse_args()

    # Check sanity of arguments
    assert args.epsilon_decay_span > 0, "EPSILON_DECAY_SPAN must be greater than 0!"
    assert args.epsilon_decay_span <= 1, "EPSILON_DECAY_SPAN must be 1 or smaller!"
    assert args.num_frames > 0, "NUM_FRAMES must be greater than 0!"
    assert args.batch_size > 0, "BATCH_SIZE must be greater than 0!"
    assert 0 <= args.gamma <= 1, "GAMMA must be between 0 and 1!"
    assert args.buffer_size > 0, "BUFFER_SIZE must be greater than 0!"
    assert args.buffer_size >= args.batch_size, "BUFFER_SIZE must be greater than or equal to BATCH_SIZE!"
    assert args.model_checkpoint_dir is not None, "MODEL_CHECKPOINT_DIR must not be None!"
    assert args.results_table is not None, "RESULTS_TABLE must not be None!"
    assert args.atari_game in set(available_atari_games.keys()), \
        f"ATARI_GAME must be one of the following: {available_atari_games.keys()} but is {args.atari_game}!"
    assert args.training_mode in available_training_versions.keys(), \
        f"TRAINING_MODE must be one of the following: {available_training_versions.keys()} but is {args.model_type}!"
    assert args.dqn_target_update_frequency >= 0, "DQN_TARGET_UPDATE_FREQUENCY must be positive!"
    assert args.start_learning_after >= 0, "START_LEARNING_AFTER must be positive!"

    print(args)

    # Start main with the arguments
    main(args.epsilon_decay_span, args.num_frames, args.batch_size, args.gamma, args.buffer_size,
         args.model_checkpoint_dir, args.snapshot_frequency, args.results_table, args.atari_game, args.training_mode,
         args.dqn_target_update_frequency, args.start_learning_after, args.verbose)
