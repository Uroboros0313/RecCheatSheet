import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Dropout, LayerNormalization

from core import MLP


class TransformerEncoder(Layer):
    def __init__(self, input_dim, ffn_unit, d_model=32, seq_len=0, num_heads=4, num_blocks=2, dropout_rate=0.2):
        super().__init__()

        self.input_dim = input_dim
        self.ffn_unit = ffn_unit
        self.seq_len = seq_len
        self.d_model = d_model
        self.num_blocks = num_blocks

        self.mhsa_layers = []
        self.ffn_layers = []
        self.ln_layers = []
        for i in range(num_blocks):
            in_c = input_dim if i == 0 else d_model
            self.mhsa_layers.append(MultiHeadSelfAttentionLayer(in_c,
                                                                attn_head_dim=d_model // num_heads,
                                                                seq_len=seq_len,
                                                                dropout_rate=dropout_rate,
                                                                pooling_type='concat'))
            self.ffn_layers.append(MLP([ffn_unit, d_model], act='relu', dropout_rate=dropout_rate))
            self.ln_layers.append(LayerNormalization(epsilon=1e-3))

    def call(self, x, mask=None):
        for i in range(self.num_blocks):
            x = self.mhsa_layers[i](x, mask)
            prev_x = x
            x = self.ffn_layers[i](x)
            x = self.ln_layers[i](x)
            x = prev_x + x

        return x


class MultiHeadSelfAttentionLayer(Layer):
    def __init__(self,
                 input_dim,
                 attn_head_dim,
                 seq_len=0,
                 num_heads=4,
                 dropout_rate=0.2,
                 pos_emb='fixed',
                 pooling_type='concat'):
        super().__init__()

        self.input_dim = input_dim
        self.attn_head_dim = attn_head_dim
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.pooling_type = pooling_type
        self.pos_emb = pos_emb

    def _generate_fixed_position_embedding_value(self):
        pos_idx = np.arange(self.seq_len).reshape(1, self.seq_len, 1)
        dim_idx = np.arange(self.input_dim).reshape(1, 1, self.input_dim)

        angle_rates = 1 / np.power(10000, 2 * (dim_idx // 2) / self.input_dim)
        angle_rads = pos_idx * angle_rates

        return angle_rads

    def build(self, input_shape):
        all_heads_attn_dim = self.attn_head_dim * self.num_heads
        self.Q = self.add_weight(shape=[self.input_dim, all_heads_attn_dim], name=f'{self.__class__}_Q',
                                 dtype=tf.float32)
        self.K = self.add_weight(shape=[self.input_dim, all_heads_attn_dim], name=f'{self.__class__}_K',
                                 dtype=tf.float32)
        self.V = self.add_weight(shape=[self.input_dim, all_heads_attn_dim], name=f'{self.__class__}_V',
                                 dtype=tf.float32)

        if self.pos_emb == 'dynamic':
            self.PE = self.add_weight(shape=[1, self.seq_len, self.input_dim])
        elif self.pos_emb == 'fixed':
            angle_rads = self._generate_fixed_position_embedding_value()
            angle_rads[:, :, 0::2] = np.sin(angle_rads[:, :, 0::2])
            angle_rads[:, :, 1::2] = np.cos(angle_rads[:, :, 1::2])
            self.PE = tf.cast(angle_rads, tf.float32)
        else:
            self.PE = tf.zeros(shape=[1, self.seq_len, self.input_dim], dtype=tf.float32)

        self.dropout = Dropout(self.dropout_rate)

    def attention_head_transform(self, mat):
        seq_len = mat.get_shape().as_list()[1]
        mat = tf.reshape(mat, shape=[-1, seq_len, self.num_heads, self.attn_head_dim])
        mat = tf.transpose(mat, perm=[0, 2, 1, 3])
        return mat

    def scaled_dot_product_attention_probs(self, queries, keys, mask):
        attn_scores = tf.matmul(queries, tf.transpose(keys, (0, 1, 3, 2)))
        attn_scores = attn_scores / tf.sqrt(tf.constant(self.attn_head_dim, dtype=tf.float32))

        if mask == None:
            mask = tf.zeros_like(attn_scores, dtype=tf.float32)
        else:
            mask = (1.0 - mask) * -1e4

        attn_probs = tf.nn.softmax(attn_scores + mask, axis=-1)
        return attn_probs

    def call(self, x, mask=None):
        shape_ = x.get_shape().as_list()
        seq_len, input_dim = shape_[1], shape_[2]

        x = x + self.PE
        queries = self.attention_head_transform(tf.matmul(x, self.Q))
        keys = self.attention_head_transform(tf.matmul(x, self.K))
        values = self.attention_head_transform(tf.matmul(x, self.V))

        attn_probs = self.scaled_dot_product_attention_probs(queries, keys, mask)
        attn_probs = self.dropout(attn_probs)
        context = tf.transpose(tf.matmul(attn_probs, values), perm=[0, 2, 3, 1])

        if self.pooling_type == 'mean':
            attn_out = tf.reduce_mean(context, dim=-1)
        elif self.pooling_type == 'concat':
            attn_out = tf.reshape(context,
                                  shape=[-1, seq_len, self.num_heads * self.attn_head_dim])

        return attn_out


if __name__ == '__main__':
    def test_mhsa():
        enc = TransformerEncoder(input_dim=128,
                                 ffn_unit=16,
                                 d_model=64,
                                 seq_len=20,
                                 num_heads=4)
        mock_input = tf.constant(np.random.randn(64, 20, 128), dtype=tf.float32)
        print(enc(mock_input))


    test_mhsa()
