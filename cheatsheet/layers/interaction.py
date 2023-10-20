import tensorflow as tf
from tensorflow.keras.initializers import glorot_normal, glorot_uniform, zeros
from tensorflow.keras.layers import Layer, Dense
from tensorflow.keras.regularizers import l2

from activations import build_act_fn


class FM(Layer):
    def __init__(self):
        super().__init__()

    def call(self, x):
        '''
        param x: input, shape [batch_size, num_feat, emb_dim]
        '''
        sum_of_square = tf.reduce_sum(tf.math.pow(x, 2), axis=1, keepdims=True)
        square_of_sum = tf.math.pow(tf.reduce_sum(x, axis=1, keepdims=True), 2)

        out = tf.reduce_sum(0.5 * (square_of_sum - sum_of_square), axis=-1, keepdims=False)

        return out


class CrossNet(Layer):
    def __init__(self, n_cross=2, cross_type='vector', reg_w=0.0, reg_b=0.0):
        super().__init__()
        self.cross_type = cross_type
        self.n_cross = n_cross
        self.reg_w = reg_w
        self.reg_b = reg_b

        self.w = []
        self.b = []

    def build(self, input_shape):
        emb_dim = input_shape[1]

        if self.cross_type == 'vector':
            for i in range(self.n_cross):
                self.w.append(Dense(1,
                                    activation=None,
                                    use_bias=False,
                                    kernel_initializer=glorot_normal(),
                                    kernel_regularizer=l2(self.reg_w))
                              )

                self.b.append(self.add_weight(name=f'vec_b_{i}',
                                              shape=(emb_dim),
                                              initializer=zeros(),
                                              regularizer=l2(self.reg_b),
                                              trainable=True)
                              )
        elif self.cross_type == 'matrix':
            for i in range(self.n_cross):
                self.w.append(Dense(emb_dim,
                                    activation=None,
                                    use_bias=False,
                                    kernel_initializer=glorot_normal(),
                                    kernel_regularizer=l2(self.reg_w)))

                self.b.append(self.add_weight(name=f'mat_b_{i}',
                                              shape=(emb_dim),
                                              initializer=zeros(),
                                              regularizer=l2(self.reg_b),
                                              trainable=True)
                              )
        else:
            raise ValueError(f"cross type `{self.cross_type}` do not exist")

    def call(self, x):
        '''
        param x: input, shape [batch_size, emb_dim]
        '''
        out = x
        for i in range(self.n_cross):
            out = out * self.w[i](out) + self.b[i] + out

        return out


class CIN(Layer):
    def __init__(self, hidden_units=[16, 32], activation='relu', reg_w=0.0):
        super().__init__()
        self.hidden_units = hidden_units
        self.activation = activation
        self.reg_w = reg_w

        self.field_nums = []
        self.cin_W = dict()
        self.cin_bias = dict()
        self.cin_acts = dict()

    def build(self, input_shape):
        input_field_num = input_shape[1]

        self.field_nums.append(input_field_num)
        self.field_nums.extend(list(self.hidden_units))

        for idx, unit in enumerate(self.hidden_units):
            self.cin_W[f'cin_W_{idx}'] = self.add_weight(name='filter' + str(idx),
                                                         shape=[1, self.field_nums[idx] * self.field_nums[0], unit],
                                                         dtype=tf.float32, initializer=glorot_uniform(),
                                                         regularizer=l2(self.reg_w))

            self.cin_bias[f'cin_bias_{idx}'] = self.add_weight(name='bias' + str(idx),
                                                               shape=[unit],
                                                               dtype=tf.float32,
                                                               initializer=zeros())
            self.cin_acts[f'cin_act_{idx}'] = build_act_fn(self.activation)

    def call(self, x):
        emb_dim = x.shape[-1]
        hidden_layer_outputs = [x]
        tensor_input = tf.transpose(tf.expand_dims(x, 2), perm=[0, 3, 1, 2])  # [BATCH_SIZE, EMB_DIM, NUM_FIELD, 1]

        for idx, unit in enumerate(self.hidden_units):
            tensor_curr = tf.transpose(tf.expand_dims(hidden_layer_outputs[-1], 2),
                                       perm=[0, 3, 2, 1])  # [BATCH_SIZE, EMB_DIM, 1, NUM_FIELD]

            tensor_inter = tf.reshape(tensor=tf.matmul(tensor_input, tensor_curr),
                                      shape=[-1, emb_dim, self.field_nums[0] * self.field_nums[idx]])  # [BATCH_SIZE, EMB_DIM, INPUT_NUM_FEAT * CUR_NUM_FEAT]

            tensor_next = tf.nn.conv1d(input=tensor_inter, filters=self.cin_W[f'cin_W_{idx}'], stride=1,
                                       padding='VALID', data_format='NWC')  # [BATCH_SIZE, EMB_DIM, NEXT_NUM_FEAT]

            tensor_next = tf.nn.bias_add(tensor_next, self.cin_bias[f'cin_bias_{idx}'])

            tensor_next = self.cin_acts[f'cin_act_{idx}'](tensor_next)

            tensor_next = tf.transpose(tensor_next, perm=[0, 2, 1])

            hidden_layer_outputs.append(tensor_next)

        final_outs = hidden_layer_outputs[1:]
        out = tf.concat(final_outs, axis=1)  # [BATCH_SIZE, H_1 + ... + H_K, EMB_DIM]
        out = tf.reduce_sum(out, axis=-1)  # [BATCH_SIZE, EMB_DIM]

        return out


if __name__ == '__main__':
    import numpy as np

    x_cn = tf.cast(np.random.randn(64, 5120), dtype=tf.float32)
    x_fm = tf.cast(np.random.randn(64, 10, 32), dtype=tf.float32)
    x_cin = tf.cast(np.random.randn(64, 12, 32), dtype=tf.float32)

    vcn = CrossNet(n_cross=2, cross_type='vector')
    mcn = CrossNet(n_cross=2, cross_type='matrix')
    fm = FM()
    cin = CIN()
    print(f"CrossNet Vector: {tf.shape(vcn(x_cn))}")
    print(f"CrossNet Matrix: {tf.shape(mcn(x_cn))}")
    print(f"FM: {tf.shape(fm(x_fm))}")
    print(f"CIN: {tf.shape(cin(x_cin))}")
