import tensorflow as tf
import numpy as np
import random

from Q_network import DQN
from buffer_DQN import ExperienceReplayBuffer


class DeepQAgent:

    def __init__(
        self, 
        action_size, 
        state_size, 
        epsilon, 
        epsilon_min, 
        epsilon_decay,
        brain, 
        buffer_size,
        batch_size,
        episodes,
        gamma
        ):

        # set action and state size
        self.action_size = action_size
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.decay = epsilon_decay
        self.brain = brain
        self.batch_size = batch_size
        self.episodes = episodes
        self.gamma = gamma

        # set random seeds
        # initialize agent hyperparams

        # initialize buffer
        self.memory = ExperienceReplayBuffer(capacity=buffer_size)

        # initialize q-network
        self.q_network = DQN(state_size, action_size)
        self.target_network = DQN(state_size, action_size)

        # update target network
        self.update_target("hard")

        # set optimizer and loss
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.loss = tf.losses.MeanSquaredError()

    def choose_action(self, observation):
        if not self.sufficient_experience:
            action = self.brain.action_spec.random_action(self.action_size)
        else:
            action = self.get_epsilon_greedy_action(observation)
        return action

    def get_epsilon_greedy_action(self, state):
        if random.random() < self.epsilon:
            action = np.random.randint(5)
            action = np.expand_dims(action, -1)
            action = np.expand_dims(action, -1)
        else:
            # TODO: no gradient?
            action_values = self.q_network(state).numpy()
            action = np.argmax(action_values, axis=1)
            action = np.expand_dims(action, -1)
        return action

    def sufficient_experience(self):
        """Returns true once the replay buffer has a certain length."""
        if len(self.memory) == self.batch_size * 50:
            print('Heureka, it is warm!')
        return len(self.memory) >= self.batch_size * 50

    def learn(self):
        if not self.sufficient_experience():
                return
        
        loss = -1
        for _ in range(self.episodes):

            # sample trajectories from replay buffer
            observations, actions, rewards, next_observations, dones = self.memory.sample(self.batch_size)

            # TODO: outsource to buffer
            observations = np.array(observations)
            actions = np.array(actions)
            rewards = np.array(rewards)
            next_observations = np.array(next_observations)
            dones = np.array(dones)

            with tf.GradientTape() as tape:

                # get max predicted Q values (for next states) from target model
                next_q_values = tf.stop_gradient(tf.math.reduce_max(self.target_network(next_observations))) # axis=1
                q_values = tf.stop_gradient(tf.math.reduce_max(self.target_network(observations)))

                # calculate the target values Q' with the Bellman equation (reward + gamma * q')
                # targets = (reward + (1.0 - done) * self.gamma * next_q_values)
                targets = tf.math.add(rewards, tf.math.multiply(self.gamma, next_q_values, (1 - dones)))

                td_error = targets - q_values

                # get expected q-values/predictions and compute MSE between target and expected
                predictions =  tf.gather_nd(
                    self.q_network(observations), # predictions 
                    tf.stack([tf.range(self.batch_size), tf.cast(tf.squeeze(actions), tf.int32)], axis=1)) # indices

                loss = self.loss(predictions, targets)

            # calculate gradients
            gradients = tape.gradient(loss, self.q_network.trainable_variables)
            self.optimizer.apply_gradients((zip(gradients, self.q_network.trainable_variables)))

            # self._soft_update_target_q_network_parameters()
        
        return (loss, td_error, predictions)

    def update_target(self, mode):
        if mode == "hard":
            self.target_network.set_weights(self.q_network.get_weights())
        else:
            print("Update mode invalid! Must be 'hard'")

    def decay_epsilon(self):
        # in_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay * episode)
        # self.epsilon = 0.01 + (self.max_epsilon - 0.01) * np.exp(-self.decay * episode)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.decay