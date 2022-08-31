import tensorflow as tf
import keras.layers as layers


def initialize_weights():
    pass

class Actor(tf.keras.Model):
    """
    The actor takes a state and outputs an estimated best action.
    Implemented as simple convolutional network to process visual inputs.
    """

    def __init__(self, action_size):
        super(Actor, self).__init__()

        # initialize weights 

        # define layers
        self.actor_layers = [
            layers.Dense(units=300, activation=tf.nn.relu),
            layers.Dense(units=200, activation=tf.nn.relu),
            layers.Dense(units=action_size, activation=tf.nn.softmax)]
        # output: softmax for discrete actions; tanh or other for continuous

    @tf.function
    def call(self, x, training=True):
        for layer in self.actor_layers:
            x = layer(x)
        return x



class Critic(tf.keras.Model):
    """
    The critic takes the states and actions (?!) from the agent and outputs a probability 
    distribution over estimated Q-values.

    In the multi-agent case, the critic takes the states and actions from BOTH agents, 
    performing so-called centralized training.
    """

    def __init__(self):
        super(Critic, self).__init__()

        # also convolutional model for processing the states?
        self.critic_layers = []
        # transform to Q_probs and apply softmax to get actual probabilities

    @tf.function
    def call(self, x, training):

        for layer in self.conv_layers:
            if (isinstance(layer, tf.keras.layers.BatchNormalization)):
                x = layer(x, training)
            else:
                x = layer(x)

        return x
