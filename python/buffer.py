import numpy as np
import random 
from collections import deque

class ReplayBuffer:

    def __init__(self, size, n_steps, discount_rate):
        self.size = size
        self.n_steps = n_steps
        self.discount_rate = discount_rate

        # holds all the experience to be sampled for training
        self.deque = deque(maxlen=self.size)
        # new experience first goes here until n timesteps have passed
        self.n_step_deque = deque(maxlen=self.n_steps)

    def push(self, transition):
        """Adds transition to the buffer."""

        self.n_step_deque.append(transition)
        if len(self.n_step_deque) == self.n_steps:
            self.deque.append(self.get_bootstrapped_transition())   

    def get_bootstrapped_transition(self):
        start_obs, start_action, start_reward, start_next_obs, start_done = self.n_step_deque[0]
        n_obs, n_action, n_reward, n_next_obs, n_done = self.n_step_deque[-1]

        summed_reward = np.zeros(2)
        for i, n_transition in enumerate(self.n_step_deque):
            obs, action, reward, next_obs, done = n_transition
            summed_reward += reward * self.discount_rate**(i+1)

            if np.any(done):
                break

        return [start_obs, start_action, summed_reward, n_next_obs, n_done]

    def sample(self, batch_size):
        """Samples a batch from the replay buffer."""
        samples = random.sample(self.deque, batch_size)
        return transpose_list(samples)

    def __len__(self):
        return len(self.deque)

    def reset(self):
        """Resets n-step deque between episodes."""
        self.n_step_deque = deque(maxlen=self.n_steps)
