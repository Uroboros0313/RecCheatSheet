import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import glorot_normal

from activations import build_act_fn


class SENet(Layer):
    def __init__(self, act='relu', ratio=4):
        super().__init__()

        self.act = act
        self.ratio = ratio

        self.input_fc = None
        self.act_fn = None
        self.output_fc = None

    def build(self, input_shape):
        num_feats = input_shape[1]

        self.input_fc = Dense(units=num_feats // self.ratio)
        self.act_fn = build_act_fn(self.act)
        self.output_fc = Dense(units=num_feats)

    def call(self, x):
        '''
        param x: input data x, shape (BATCH_SIZE, NUM_FEATS, EMB_DIM)
        '''
        w_0 = tf.reduce_mean(x, axis=-1)
        w_1 = self.input_fc(w_0)
        w_2 = self.act_fn(w_1)
        w_3 = self.output_fc(w_2)
        print(w_0.shape, w_1.shape, w_2.shape, w_3.shape)

        w = tf.expand_dims(tf.nn.sigmoid(w_3) * 2, axis=2)
        return x * w


class CancelOut(Layer):
    def __init__(self):
        super().__init__()


class DropRank(Layer):
    def __init__(self):
        super().__init__()


if __name__ == "__main__":
    import numpy as np


    def test_senet():
        mock_ninput = tf.constant(np.random.randn(64, 32, 10), dtype=tf.float32)
        se = SENet()
        print(se(mock_ninput))


    test_senet()
