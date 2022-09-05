import collections
import random
import tensorflow as tf
import numpy as np

class ExperienceReplayBuffer():

    def __init__(self, capacity):
        """
        Experience replay buffer with a max capacity of the capacity parameter. FIFO 
        style.
        """

        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)
 
    def push(self, item):
        self.buffer.append(item)

    def sample(self, batch_size):

        # Sample minibatch from replay buffer
        # minibatch = []
        # for _ in np.arange(batch_size):
        #   minibatch.append(self.buffer.pop()) # TODO could move replay buffer to class and create minibatch function
        minibatch = random.sample(self.buffer, batch_size)

        # workaround because slicing is not working
        s, a, r, s_, d = [], [], [], [], []
        for m in np.array(minibatch[:]):
            s.append(m[0])
            a.append(m[1])
            r.append(m[2])
            s_.append(m[3])
            d.append(m[4])
        
        # s = tf.convert_to_tensor(s)
        # a = tf.convert_to_tensor(a)
        # r = tf.cast(tf.convert_to_tensor(r), dtype=tf.float32)
        # s_ = tf.convert_to_tensor(s_)

        return s, a, r, s_, d