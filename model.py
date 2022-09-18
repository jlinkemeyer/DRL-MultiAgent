import tensorflow as tf


class DQN(tf.keras.Model):

    def __init__(self, action_size):
        """
        A simple Deep Q network with two dense layers with 64 units each, and an output layer with an output for each
        possible action.
        :param action_size: number of possible actions
        """
        super(DQN, self).__init__()

        self.DQN_layers = [
            # tf.keras.layers.InputLayer(input_shape=state_size),
            tf.keras.layers.Dense(units=64, activation='relu'),
            tf.keras.layers.Dense(units=64, activation='relu'),
            tf.keras.layers.Dense(units=action_size, activation='linear')
        ]
    
    @tf.function
    def call(self, x):
        """
        Network call function. Passes an input x through the network.
        :param x: network input
        :return: network output
        """
        for layer in self.layers:
            x = layer(x)
        return x
