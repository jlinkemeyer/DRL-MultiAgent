import tensorflow as tf
import numpy as np

from DQN_agent import DeepQAgent

class DoubleDeepQAgent(DeepQAgent):

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
        gamma,
        alpha,
        batch_factor,
        lr_decay_steps,
        lr_decay_rate
        ):
        super(DoubleDeepQAgent, self).__init__(
            action_size, 
            state_size, 
            epsilon, 
            epsilon_min, 
            epsilon_decay,
            brain, 
            buffer_size, 
            batch_size, 
            episodes, 
            gamma,
            alpha,
            batch_factor,
            lr_decay_steps,
            lr_decay_rate
        )

    def learn(self, incr_batch, decr_lr):
        if not self.sufficient_experience():
                return

        if incr_batch:
            self.batch_size = self.batch_size * self.batch_factor
        
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

                q_next_target = tf.stop_gradient(self.target_network(next_observations))
                q_next_main = tf.stop_gradient(self.q_network(next_observations))

                # use main network to choose max action of next state
                max_actions = np.argmax(q_next_main, axis=1)

                # gather target network q-values of actions selected by main network
                q_next =  tf.gather_nd(
                    q_next_target, # predictions 
                    tf.stack([tf.range(self.batch_size), tf.cast(max_actions, tf.int32)], axis=1)) # indices

                # get q-values from target network (at the indices chosen by main network)
                targets = tf.math.add(rewards, tf.math.multiply(self.gamma, q_next, (1 - dones)))
                # td_error = targets - q_values

                # get expected q-values/predictions and compute MSE between target and expected
                predictions =  tf.gather_nd(
                    self.q_network(observations), # predictions 
                    tf.stack([tf.range(self.batch_size), tf.cast(tf.squeeze(actions), tf.int32)], axis=1)) # indices

                loss = self.loss(predictions, targets)

            # calculate gradients
            gradients = tape.gradient(loss, self.q_network.trainable_variables)
            
            if decr_lr:
                lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                                self.alpha,
                                decay_steps=self.lr_decay_steps,
                                decay_rate=self.lr_decay_rate,
                                staircase=True)
                optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
            else:
                optimizer = tf.keras.optimizers.Adam(learning_rate=self.alpha)
            optimizer.apply_gradients((zip(gradients, self.q_network.trainable_variables)))

            # self._soft_update_target_q_network_parameters()
        
        return (loss, None, predictions)



    