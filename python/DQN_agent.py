import tensorflow as tf
import numpy as np
import random

from model import DQN
from memory import ExperienceReplayBuffer


class DeepQAgent:

    def __init__(self, action_size, epsilon, epsilon_min, epsilon_decay, brain, buffer_size, batch_size,
                 epochs, gamma, alpha, batch_factor, lr_decay_steps, lr_decay_rate, decr_lr):

        # Set agent hyperparameters
        self.action_size = action_size
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.decay = epsilon_decay
        self.brain = brain
        self.batch_size = batch_size
        self.epochs = epochs
        self.gamma = gamma
        self.alpha = alpha
        self.batch_factor = batch_factor
        self.lr_decay_steps = lr_decay_steps
        self.lr_decay_rate = lr_decay_rate
        self.decr_lr = decr_lr

        # Initialize experience buffer
        self.memory = ExperienceReplayBuffer(capacity=buffer_size)

        # Initialize q-network
        self.q_network = DQN(action_size=action_size)
        self.target_network = DQN(action_size=action_size)
        self.update_target("hard") # TODO: Is necessary?

        # Use either a decaying learning rate or a set learning rate, depending on decr_lr (decrease learning rate)
        # flag
        if self.decr_lr:
            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                self.alpha,
                decay_steps=self.lr_decay_steps,
                decay_rate=self.lr_decay_rate,
                staircase=True
            )
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        else:
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.alpha)

        # Set loss function
        self.loss = tf.losses.MeanSquaredError()

        # define checkpoint and load (if one exists)
        self.checkpoint = tf.train.Checkpoint(**dict(q_network=self.q_network, target_network=self.target_network))
        self.manager = tf.train.CheckpointManager(self.checkpoint, './tf_checkpoints', max_to_keep=3)
        self.try2load_checkpoint()

    def try2load_checkpoint(self):
        """
        Try to load network weights from a checkpoint (called once in the beginning). User log to track whether
        checkpoint is used to initialize networks
        """
        try:
            self.checkpoint.restore(self.manager.latest_checkpoint)
            print("Restored from {}".format(self.manager.latest_checkpoint))
        except:
            print("Initializing from scratch.")

    def choose_action(self, observation):
        """
        Selects an action based on an observation. Actions are random until the replay buffer has enough samples. Then,
        actions are chosen in an epsilon-greedy manner.

        :param observation: a scene observation
        :return: an action
        """
        if not self.sufficient_experience:
            action = self.brain.action_spec.random_action(self.action_size)
        else:
            action = self.get_epsilon_greedy_action(observation)
        return action

    def get_epsilon_greedy_action(self, obs):
        """
        Returns an action based on an observation.

        :param obs: an observation from the scene
        :return: an epsilon-greedy action
        """
        if random.random() < self.epsilon:
            action = np.random.randint(5)
            action = np.expand_dims(action, -1)
            action = np.expand_dims(action, -1)
        else:
            action_values = self.q_network(obs).numpy()
            action = np.argmax(action_values, axis=1)
            action = np.expand_dims(action, -1)
        return action

    def sufficient_experience(self):
        """
        Checks whether the replay buffer has a specific length.

        :return: true if replay buffer has more than a specific number of items
        """
        # TODO delete
        if len(self.memory) == self.batch_size * 50:
            print('Heureka, it is warm!')
        return len(self.memory) >= self.batch_size * 50

    def learn(self):
        """
        Deep Q-learning network learning function. Updates network weights based on samples from the experience replay
        buffer.

        :return: loss from training step
        """
        # Sample experiences from replay buffer
        observations, actions, rewards, next_observations, dones = self.memory.sample(self.batch_size)

        with tf.GradientTape() as tape:

            # Get max predicted Q values (for next states) from target model
            next_q_values = tf.stop_gradient(tf.math.reduce_max(self.target_network(next_observations)))

            # Calculate the target values Q' with the Bellman equation (reward + gamma * q'), i.e.
            # targets = (reward + (1.0 - done) * self.gamma * next_q_values)
            targets = tf.math.add(rewards, tf.math.multiply(self.gamma, next_q_values, (1 - dones)))

            # Get expected q-values/predictions and compute MSE between target and prediction
            predictions = tf.gather_nd(
                self.q_network(observations), # predictions
                tf.stack([tf.range(self.batch_size), tf.cast(tf.squeeze(actions), tf.int32)], axis=1)) # indices

            loss = self.loss(predictions, targets)

        # Calculate gradients and update Q network model weights
        gradients = tape.gradient(loss, self.q_network.trainable_variables)
        self.optimizer.apply_gradients((zip(gradients, self.q_network.trainable_variables)))
        
        return loss

    def update_target(self, mode):
        """
        Updates the target network given a mode. Currently only hard updates are possible, i.e. the target network fully
        takes the weights from the Q network.
        :param mode: update mode
        """
        if mode == "hard":
            self.target_network.set_weights(self.q_network.get_weights())
        else:
            print("Update mode invalid! Must be 'hard'")

    def decay_epsilon(self):
        """
        Epsilon decay function. Over time, epsilon can be schedules to decay to decrease random exploration.
        """
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.decay

    def get_learning_rate(self):
        """
        Returns the current learning rate. When learning rate decay is used, this function can be called to track the
        learning rate decay.
        :return: current learning rate
        """
        if self.decr_lr:
            return self.optimizer._decayed_lr(tf.float32)
        else:
            return self.optimizer._lr(tf.float32)

    def get_batch_size(self):
        """
        Returns the current batch size. Helpful when batch size increase is performed in learning steps.
        """
        return self.batch_size

    def increase_batch_size(self):
        """
        Function to increase the batch size with a given factor over time.
        """
        self.batch_size = self.batch_size * self.batch_factor
