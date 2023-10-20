import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, BatchNormalization, Dropout


class MLP(Layer):
    def __init__(self, hidden_units, act='relu', dropout_rate=0.0, use_bn=False):
        super().__init__()
        self.hidden_units = hidden_units
        self.act = act
        self.dropout_rate = dropout_rate
        self.use_bn = use_bn

        self.dense_layers = [Dense(units,
                                   activation=act,
                                   kernel_initializer=tf.random_normal_initializer(),
                                   kernel_regularizer=None,
                                   use_bias=True) for units in hidden_units]

        self.dropout_layers = [Dropout(rate=dropout_rate) for _ in range(len(hidden_units))]

        self.bn_layers = [BatchNormalization() for _ in range(len(hidden_units))] if use_bn else []

    def call(self, inputs):
        x = inputs
        for i in range(len(self.hidden_units)):
            x = self.dense_layers[i](x)
            x = self.dropout_layers[i](x)
            if self.use_bn:
                x = self.bn_layers[i](x)

        return x


class Linear(Layer):
    def __init__(self,
                 sigmoid=True):
        super().__init__()
        self.sigmoid = sigmoid

    def call(self, x):
        linear_logits = Dense(1, use_bias=True)(x)

        if self.sigmoid:
            linear_logits = tf.sigmoid(linear_logits)

        return linear_logits


class CapsuleNetwork(Layer):
    def __init__(self):
        super().__init__()

    def build(self, input_shape):
        pass

    def call(sefl, x):
        pass
