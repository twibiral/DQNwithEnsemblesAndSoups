import numpy as np
import tensorflow as tf


class DQNAgent:
    def __init__(self,
                 replay_buffer,
                 model: tf.keras.Model,
                 gamma: float = 0.99,
                 double_dqn: bool = False):
        """
        Initialise the DQN algorithm with the given model.
        Set use_double_dqn to True to use the double DQN algorithm by Hasselt et al. (2015)

        :param replay_buffer: storage for experience replay
        :param model: the Q-network
        :param gamma: the discount factor for the temporal difference
        :param double_dqn: whether to use double DQN
        """
        self.memory = replay_buffer
        self.gamma = tf.convert_to_tensor(gamma, dtype=tf.float32)
        self.q_network = model
        self.double_dqn = double_dqn

        # Only used for Double DQ-learning:
        self.target_network = tf.keras.models.clone_model(model) if double_dqn else None

    def optimise_td_loss(self, batch_size: int):
        """
        Optimise the TD-error over a single minibatch of transitions.

        Method calls the model via @tf.functions to speed up the execution
        :return: the loss
        """
        states, rewards, next_states, dones = self.memory.sample(batch_size)

        # Get the Q-values for the current states (Not necessary)
        # q_values = self.q_network.predict(states, verbose=0)

        # Get the Q-values for the next states
        if self.double_dqn:
            # Get the Q-values for the next states using the target network only when using double DQN
            next_q_values = self._get_model_pred(self.target_network, next_states)
        else:
            next_q_values = self._get_model_pred(self.q_network, next_states)

        # Calculate the target Q-values
        target_q_values = rewards + self.gamma * tf.reduce_max(next_q_values, axis=1) * (1 - dones)

        # Train the model on a single batch, return the loss
        loss = self.q_network.train_on_batch(states, target_q_values)

        return loss

    @tf.function(autograph=False)
    def _get_model_pred(self, model, input):
        return model(input)

    @tf.function(autograph=False)
    def act(self, state: np.ndarray, scaling_factor: float = 1/255.0):
        """
        Select an action greedily from the Q-network given the state
        :param state: the current state
        :param scaling_factor: state is scaled by this factor before being passed to the network. (Atatari uses 1/255)
        :return: the action to take
        """
        # with tf.device(None):  # Use the default device
        state = tf.cast(state, dtype=tf.float32) * scaling_factor
        state = tf.expand_dims(state, axis=0)

        q_values = self.q_network(state)
        action = tf.argmax(q_values, axis=1)
        action_item = action[0]  # Get the scalar value from the tensor

        return action_item

    def update_target_network(self):
        self.target_network = tf.keras.models.clone_model(self.q_network)
