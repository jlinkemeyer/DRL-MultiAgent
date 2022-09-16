import tensorflow as tf
import numpy as np

from DQN_agent import DeepQAgent


class DoubleDeepQAgent(DeepQAgent):

    def __init__(self, action_size, state_size, epsilon, epsilon_min, epsilon_decay, brain, buffer_size, batch_size,
                 epochs, gamma, alpha, batch_factor, lr_decay_steps, lr_decay_rate, decr_lr):

        # Initialize the double deep Q agent with the same parameters as the 'regular' deep Q agent
        super(DoubleDeepQAgent, self).__init__(
            action_size, 
            state_size, 
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

        # TODO: comments
        with tf.GradientTape() as tape:

            q_next_target = tf.stop_gradient(self.target_network(next_observations))
            q_next_main = tf.stop_gradient(self.q_network(next_observations))

            # use main network to choose max action of next state
            max_actions = np.argmax(q_next_main, axis=1)

            # gather target network q-values of actions selected by main network
            q_next = tf.gather_nd(
                q_next_target, # predictions
                tf.stack([tf.range(self.batch_size), tf.cast(max_actions, tf.int32)], axis=1)) # indices

            # get q-values from target network (at the indices chosen by main network) # TODO I dont get this comment
            targets = tf.math.add(rewards, tf.math.multiply(self.gamma, q_next, (1 - dones)))

            # get expected q-values/predictions and compute MSE between target and expected
            predictions = tf.gather_nd(
                self.q_network(observations), # predictions
                tf.stack([tf.range(self.batch_size), tf.cast(tf.squeeze(actions), tf.int32)], axis=1)) # indices

            loss = self.loss(predictions, targets)

        # calculate gradients
        gradients = tape.gradient(loss, self.q_network.trainable_variables)
        self.optimizer.apply_gradients((zip(gradients, self.q_network.trainable_variables)))
        
        return loss
