import tensorflow as tf
from tensorflow.keras.layers import Layer

from core import MLP, Linear


class MVKEBlock(Layer):
    def __init__(self,
                 num_feats,
                 num_kernel=4,
                 hidden_dims=[32, 16],
                 emb_dim=32,
                 combiner='weighted_sum',
                 dropout_rate=0.3):
        super().__init__()

        self.num_feats = num_feats
        self.num_kernel = num_kernel
        self.hidden_dims = hidden_dims
        self.emb_dim = emb_dim
        self.combiner = combiner
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        self.vk_embs = self.add_weight(name="w",
                                       shape=(self.emb_dim, self.num_kernel),
                                       trainable=True)
        '''
        if self.combiner == 'weighted_sum':
            input_dim = self.emb_dim
        elif self.combiner == 'concat':
            input_dim = self.emb_dim * self.num_feats
        '''

        self.vk_towers = [MLP(self.hidden_dims,
                              act='relu',
                              dropout_rate=self.dropout_rate,
                              use_bn=True) for _ in range(self.num_kernel)]

    def virtual_kernel_expert_embeddings(self, x):
        vk_scores = tf.matmul(x, self.vk_embs)
        vk_probs = tf.nn.softmax(tf.transpose(vk_scores, perm=(0, 2, 1)), axis=-1)
        vk_probs = tf.nn.dropout(vk_probs, rate=self.dropout_rate)

        if self.combiner == 'weighted_sum':
            context = tf.matmul(vk_probs, x) / tf.sqrt(tf.constant(self.emb_dim, dtype=tf.float32))
        elif self.combiner == 'concat':
            vk_probs = tf.expand_dims(vk_probs, axis=3)
            x = tf.expand_dims(x, axis=1)
            context = tf.reshape(tf.multiply(vk_probs, x),
                                 (-1, self.num_kernel, self.num_feats * self.emb_dim))
        return context

    def virtual_kernel_gate_probs(self, tag_embedding):
        gate_scores = tf.matmul(tag_embedding, self.vk_embs) / tf.sqrt(tf.constant(self.emb_dim, dtype=tf.float32))
        gate_probs = tf.expand_dims(tf.nn.softmax(gate_scores, axis=-1), axis=-1)
        gate_probs = tf.nn.dropout(gate_probs, rate=self.dropout_rate)

        return gate_probs

    def virtual_kernel_dnn_outputs(self, context):
        vk_outs = \
            [tf.expand_dims(self.vk_towers[i](context[:, i, :]), axis=1) for i in range(self.num_kernel)]
        vk_outs = tf.concat(vk_outs, axis=1)
        return vk_outs

    def call(self, x, tag_embedding):
        context = self.virtual_kernel_expert_embeddings(x)
        vk_outs = self.virtual_kernel_dnn_outputs(context)
        gate_probs = self.virtual_kernel_gate_probs(tag_embedding)

        user_embedding = \
            tf.reduce_sum(tf.multiply(vk_outs, gate_probs), axis=1)

        return user_embedding


if __name__ == "__main__":
    import numpy as np

    a = tf.Variable(np.random.randn(20, 30), dtype=tf.float32)
    b = tf.Variable(np.random.randn(30, 40), dtype=tf.float32)
    print(a / tf.math.sqrt(tf.constant(12, dtype=tf.float32)))
    print(tf.transpose(tf.matmul(a, b), perm=(0, 1)))
    print(a.get_shape().as_list())


    def test_mvke():
        mock_ninput = tf.Variable(np.ones(shape=(64, 10, 32)), dtype=tf.float32)
        mock_tag = tf.Variable(np.random.randn(64, 32), dtype=tf.float32)
        model = MVKEBlock(10, combiner='concat')
        print(model(mock_ninput, mock_tag))


    test_mvke()
