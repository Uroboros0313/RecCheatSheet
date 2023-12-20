import tensorflow as tf
from tensorflow.keras.layers import Layer, Dropout, Dense, BatchNormalization

from core import MLP


class ResidualCrossTower(Layer):
    def __init__(self,
                 output_dim=32,
                 act='relu',
                 combiner='mean',
                 hidden_units=[128, 64],
                 dropout_rate=0.3):
        super().__init__()
        self.output_dim = output_dim
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate
        self.act = act
        self.combiner = combiner

        self.dense_layers = []
        self.bn_layers = []
        self.dropout_layers = []
        self.skip_layers = []

    def build(self, input_shape):
        for i in range(len(self.hidden_units)):
            self.dense_layers.append(Dense(self.hidden_units[i],
                                           activation=self.act,
                                           kernel_initializer=tf.random_normal_initializer(),
                                           kernel_regularizer=None,
                                           use_bias=True))
            self.dropout_layers.append(Dropout(rate=self.dropout_rate))
            self.bn_layers.append(BatchNormalization())
            self.skip_layers.append(Dense(self.output_dim,
                                          activation=None,
                                          kernel_initializer=tf.random_normal_initializer(),
                                          use_bias=False))

        self.dense_layers.append(Dense(self.output_dim,
                                       activation=None,
                                       kernel_initializer=tf.random_normal_initializer(),
                                       kernel_regularizer=None,
                                       use_bias=True))

    def call(self, x):
        residual_lst = []
        x_hid = x

        for i in range(len(self.hidden_units)):
            x_hid = self.bn_layers[i](x_hid)

            x_res = self.skip_layers[i](x_hid)
            residual_lst.append(x_res)

            x_hid = self.dropout_layers[i](x_hid)
            x_hid = self.dense_layers[i](x_hid)

        x_end = self.dense_layers[-1](x_hid)
        residual_lst.append(x_end)
        x_con = tf.reduce_mean(residual_lst, axis=0)

        return x_con, x_end


class MMOEBlock(Layer):
    def __init__(self,
                 num_expert,
                 num_task,
                 emb_dim=32,
                 expert_hidden_units=[128, 64],
                 gate_hidden_units=[32, 16],
                 dropout_rate=0.3):
        super().__init__()

        self.num_expert = num_expert
        self.num_task = num_task
        self.emb_dim = emb_dim
        self.dropout_rate = dropout_rate
        self.expert_hidden_units = expert_hidden_units
        self.gate_hidden_units = gate_hidden_units

        self.gate_net = None
        self.expert_net = []

    def build(self, input_shape):
        self.gate_net = [MLP(hidden_units=self.gate_hidden_units + [self.num_expert],
                             dropout_rate=self.dropout_rate, use_bn=True) for _ in range(self.num_task)]

        self.expert_net = [MLP(hidden_units=self.expert_hidden_units + [self.emb_dim],
                               dropout_rate=self.dropout_rate, use_bn=True) for _ in range(self.num_expert)]

        self.gate_dropout = Dropout(rate=self.dropout_rate, )

    def call(self, x):
        '''
        param x: input shape [batch_size, concat_emb_dim]
        '''
        expert_out = [tf.expand_dims(net(x), axis=2) for net in self.expert_net]
        expert_out = tf.concat(expert_out, axis=2)  # [BATCH_SIZE, EMB_SIZE, NUM_EXPERT]

        gate_out = [tf.expand_dims(net(x), axis=2) for net in self.gate_net]
        gate_out = tf.concat(gate_out, axis=2)  # [BATCH_SIZE, NUM_EXPERT, NUM_TASK]

        gate_out = self.gate_dropout(gate_out)
        norm_gate_out = tf.nn.softmax(gate_out, axis=1)

        task_embs = tf.matmul(expert_out, norm_gate_out)
        task_embs = tf.split(task_embs, num_or_size_splits=self.num_task, axis=2)

        return task_embs


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


    # print(a / tf.math.sqrt(tf.constant(12, dtype=tf.float32)))
    # print(tf.transpose(tf.matmul(a, b), perm=(0, 1)))
    # print(a.get_shape().as_list())

    def test_mvke():
        mock_ninput = tf.Variable(np.ones(shape=(64, 10, 32)), dtype=tf.float32)
        mock_tag = tf.Variable(np.random.randn(64, 32), dtype=tf.float32)
        model = MVKEBlock(10, combiner='concat')
        # print(model(mock_ninput, mock_tag))


    def test_mmoe():
        mock_input = tf.constant(np.ones(shape=(64, 1280)), dtype=tf.float32)
        model = MMOEBlock(6, 3)
        # print(model(mock_input))


    def test_rit():
        mock_input = tf.constant(np.ones(shape=(64, 1280)), dtype=tf.float32)
        model = ResidualCrossTower()
        print(model(mock_input)[0].shape, model(mock_input)[1].shape)


    test_mvke()
    test_mmoe()
    test_rit()
