import tensorflow as tf


class DQN(tf.keras.Model):

    # TODO normalize input?

    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()

        self.DQN_layers = [
            # tf.keras.layers.InputLayer(input_shape=state_size),
            tf.keras.layers.Dense(units=64, activation='relu'),
            tf.keras.layers.Dense(units=64, activation='relu'),
            tf.keras.layers.Dense(units=action_size, activation='linear')
        ]
    
    @tf.function
    def call(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
