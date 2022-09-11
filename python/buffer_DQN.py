import collections
import random
import tensorflow as tf
import numpy as np

class ExperienceReplayBuffer():

    def __init__(self, capacity):
        """
        Experience replay buffer that contains (state, action, reward, next state, done) tuples. Has a maximal capacity.

        :param capacity: Maximal number of elements in the replay buffer
        """
        self.buffer = collections.deque(maxlen=capacity)
        self.experience = collections.namedtuple("Experience",
                                                 field_names=["state", "action", "reward", "next_state", "done"])

    def __len__(self):
        """
        Function to get the length of the experience replay buffer.

        :return: Length of the experience replay buffer
        """
        return len(self.buffer)
 
    def push(self, s, a, r, s_, d):
        """
        Add an item to the experience replay buffer. An item is a (state, action, reward, next state, done) tuple.
        :param s: state
        :param a: action
        :param r: reward
        :param s_: next state
        :param d: done
        :return:
        """
        item = self.experience(s, a, r, s_, d)
        self.buffer.append(item)

    def sample(self, batch_size):
        """
        Sample a minibatch from the experience replay buffer

        :param batch_size: number of (state, action, reward, next state, done) tuples
        :return:
        """

        # Sample minibatch from replay buffer
        # TODO could move replay buffer to class and create minibatch function
        minibatch = random.sample(self.buffer, batch_size)

        states = tf.cast(tf.convert_to_tensor(np.vstack([m.state for m in minibatch if m is not None])),
                         dtype=tf.float32)
        actions = tf.cast(tf.convert_to_tensor(np.vstack([m.action for m in minibatch if m is not None])),
                          dtype=tf.float32)
        rewards = tf.cast(tf.convert_to_tensor(np.vstack([m.reward for m in minibatch if m is not None])),
                          dtype=tf.float32)
        next_states = tf.cast(tf.convert_to_tensor(np.vstack([m.next_state for m in minibatch if m is not None])),
                              dtype=tf.float32)
        dones = tf.cast(tf.convert_to_tensor(np.vstack([m.done for m in minibatch if m is not None]).astype(np.uint8)),
                        dtype=tf.float32)

        return states, actions, rewards, next_states, dones