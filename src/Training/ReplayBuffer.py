import random
from collections import deque

import numpy as np
import tensorflow as tf


class ReplayBuffer:
    def __init__(self, size):
        self._storage = deque([], maxlen=size)
        self._maxsize = size

    def __len__(self):
        return len(self._storage)

    def add(self, state, reward, next_state, done):
        data = (state, reward, next_state, done)
        self._storage.append(data)

    def sample(self, batch_size):
        sample = random.sample(self._storage, batch_size)
        states, rewards, next_states, dones = list(map(list, zip(*sample)))   # transpose the sample list

        return np.array(states), np.array(rewards), np.array(next_states), np.array(dones)
        # return tf.convert_to_tensor(states, dtype=np.float32), tf.convert_to_tensor(rewards, dtype=np.float32), \
        #     tf.convert_to_tensor(next_states, dtype=np.float32), tf.convert_to_tensor(dones, dtype=np.float32)

    def is_filled(self):
        """
        Check if the buffer is filled to the maximum size.
        :return: True if filled completely, False otherwise
        """
        return len(self._storage) == self._maxsize


class PixelNormalizationBufferWrapper:
    """
    Wraps around a `ReplayBuffer` object and normalises the pixel values to the range [0, 1].
    This allows us to store the states as int8 and therefore save memory.
    """
    def __init__(self, buffer: ReplayBuffer):
        self.buffer = buffer

    def __len__(self):
        return len(self.buffer)

    def add(self, state, reward, next_state, done):
        state = np.array(state).astype(np.uint8)
        next_state = np.array(next_state).astype(np.uint8)
        self.buffer.add(state, reward, next_state, done)

    def sample(self, batch_size):
        states, rewards, next_states, dones = self.buffer.sample(batch_size)

        return tf.convert_to_tensor(states, dtype=np.float32)/255.0, rewards, \
            tf.convert_to_tensor(next_states, dtype=np.float32)/255.0, dones

    def is_filled(self):
        return self.buffer.is_filled()
