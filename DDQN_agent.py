import tensorflow as tf
import numpy as np

from DQN_agent import DeepQAgent


class DoubleDeepQAgent(DeepQAgent):

    def __init__(self, action_size, epsilon, epsilon_min, epsilon_decay, brain, buffer_size, batch_size,
                 epochs, gamma, alpha, batch_factor, lr_decay_steps, lr_decay_rate, decr_lr):

        # Initialize the double deep Q agent with the same parameters as the 'regular' deep Q agent
        super(DoubleDeepQAgent, self).__init__(
            action_size,
            epsilon, 
            epsilon_min, 
            epsilon_decay,
            brain, 
            buffer_size, 
            batch_size, 
            epochs,
            gamma,
            alpha,
            batch_factor,
            lr_decay_steps,
            lr_decay_rate,
            decr_lr
        )

    def learn(self):
        """
        Double deep Q-learning network learning function. Updates network weights based on samples from the experience
        replay buffer.

        :return: loss from training step
        """
        # Sample trajectories from replay buffer
        observations, actions, rewards, next_observations, dones = self.memory.sample(self.batch_size)

        with tf.GradientTape() as tape:

            # obtain predictions from both main q-network and target network
            q_next_target = tf.stop_gradient(self.target_network(next_observations))
            q_next_main = tf.stop_gradient(self.q_network(next_observations))

            # use main q-network to choose the action with the maximal q-value for the next observation (action selection)
            max_actions = np.argmax(q_next_main, axis=1)

            # extract q-values from the target network for the actions selected by the main q-network (action evaluation)
            q_next = tf.gather_nd(
                q_next_target, # predictions
                tf.stack([tf.range(self.batch_size), tf.cast(max_actions, tf.int32)], axis=1)) # indices

            # compute targets with r + gamma * q'
            targets = tf.math.add(rewards, tf.math.multiply(self.gamma, q_next, (1 - dones)))

            # obtain the expected predictions from the main q-network and compute MSE between target and expected
            predictions = tf.gather_nd(
                self.q_network(observations), # predictions
                tf.stack([tf.range(self.batch_size), tf.cast(tf.squeeze(actions), tf.int32)], axis=1)) # indices

            loss = self.loss(predictions, targets)

        # calculate gradients
        gradients = tape.gradient(loss, self.q_network.trainable_variables)
        self.optimizer.apply_gradients((zip(gradients, self.q_network.trainable_variables)))
        
        return loss
