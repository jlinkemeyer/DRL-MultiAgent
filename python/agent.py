import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

from network import Actor, Critic

class DDPGAgent(tf.keras.Model):

    def __init__(self, action_size):
        super(DDPGAgent, self).__init__()

        self.action_size = action_size

        # needs actor & target actor + critic and target critic
        self.actor = Actor(action_size)
        self.target_actor = None
        self.critic = None
        self.target_critic = None

        # initialize targets same as original networks

        # initialize optimizers with their respective learning rate

    def act(self, observation):
        """
        Get action from actor and optionally modify with noise.
        """
        probs = self.actor(observation)
        # action = tf.squeeze(tf.multinomial(logits=logits,num_samples=1), axis=1)
        # action = tf.squeeze(tfp.distributions.Multinomial(probs=probs).sample(1))
        probs = probs[0].numpy()
        action = np.random.choice(self.action_size, p=probs)
        action = np.expand_dims(action, -1)
        action = np.expand_dims(action, -1) # TODO: prettier
        
        return action 

    def target_act(self, observation):
        # return self.target_actor(observation)
        pass