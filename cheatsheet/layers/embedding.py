import tensorflow as tf
from tensorflow.keras.initializers import glorot_normal, random_uniform
from tensorflow.keras.layers import Layer, Embedding


class DenseEmbedding(Layer):
    def __init__(self, embedding_size=32, name=''):
        super().__init__(name=name)
        self.embedding_size = embedding_size
        self.dense_embedding = None

    def build(self, input_shape):
        self.dense_embedding = self.add_weight(shape=[1, 32],
                                               initializer=glorot_normal(),
                                               trainable=True)

    def call(self, x):
        max_rank = len(x.shape)

        x = tf.expand_dims(x, max_rank)
        out = x * self.dense_embedding

        return out


class AutoDisEmbedding(Layer):
    def __init__(self):
        super().__init__()

    def build(self, input_shape):
        pass

    def call(self, x):
        pass


class VanillaEmbedding(Layer):
    def __init__(self, vocab_size, embedding_size=32, use_hash=False, seed=[0, 1]):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.use_hash = use_hash
        self.seed = seed

        self.lookup = Embedding(input_dim=vocab_size,
                                output_dim=embedding_size,
                                embeddings_initializer=random_uniform(minval=-0.05 / embedding_size,
                                                                      maxval=0.05 / embedding_size),
                                trainable=True
                                )

    def call(self, x):
        if self.use_hash:
            x = tf.strings.to_hash_bucket_strong(x, num_buckets=self.vocab_size, key=self.seed)

        return self.lookup(x)


class HashEmbedding(Layer):
    def __init__(self):
        super().__init__()

    def build(self, input_shape):
        pass

    def call(self, x):
        pass


class CompositionalEmbedding(Layer):
    def __init__(self):
        super().__init__()

    def build(self, input_shape):
        pass

    def call(self, x):
        pass


class DenseHashEmbedding(Layer):
    def __init__(self):
        super().__init__()

    def build(self, input_shape):
        pass

    def call(self, x):
        pass


if __name__ == '__main__':
    import numpy as np

    input = tf.cast(np.random.randn(32, 10), tf.float32)
    de = DenseEmbedding(32)
    print(de(input).shape)

    string_input = tf.cast([[["21sji", '21323dd'],
                             ['12213', '12ddd']],
                            [["21sji", '21323dd'],
                             ['12213', '12ddd']]], tf.string)
    ret = tf.strings.to_hash_bucket_strong(string_input, num_buckets=20, key=[12, 24])
    print(ret)
