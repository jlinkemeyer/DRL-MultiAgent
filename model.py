import tensorflow as tf


class DQN(tf.keras.Model):

    def __init__(self, action_size):
        """
        A simple Deep Q network with two dense layers with 64 units each, and an plots layer with an plots for each
        possible action.
        :param action_size: number of possible actions
        """
        super(DQN, self).__init__()

        self.dense_1 = tf.keras.layers.Dense(units=64, activation='relu')
        self.dense_2 = tf.keras.layers.Dense(units=64, activation='relu')
        self.out = tf.keras.layers.Dense(units=action_size, activation='linear')
    
    @tf.function
    def call(self, x):
        """
        Network call function. Passes an input x through the network.
        :param x: network input
        :return: network plots
        """

        x = self.dense_1(x)
        x = self.dense_2(x)
        x = self.out(x)

        return x
