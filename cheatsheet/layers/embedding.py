import random

import tensorflow as tf
from tensorflow.keras.initializers import glorot_normal, glorot_uniform, random_uniform
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
    '''
    Hash Embeddings for Efficient Word Representations
    '''
    def __init__(self, vocab_size, embedding_size=32, num_lookup=2, seed=[0, 1], keys=[[0, 1], [2, 3]]):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.num_lookup = num_lookup
        
        self.seed = seed
        if len(self.seed) != num_lookup:
            self.seed.extend([random.randint(0, 10**3) for _ in range(num_lookup - len(seed))])
        
        self.keys = keys
        if len(self.keys) != num_lookup:
            self.keys.extend([[random.randint(0, 10**2), random.randint(0, 10**3)] 
                               for _ in range(num_lookup - len(keys))])
        
        self.lookup_list = None
        self.hash_fn_list = None
        self.lookup_weights = None
        
    def build(self, input_shape):
        self.hash_fn_list = [lambda x: tf.strings.to_hash_bucket_strong(x, 
            num_buckets=self.vocab_size, key=self.keys[i]) for i in range(self.num_lookup)]
        self.lookup_list = [Embedding(self.vocab_size, self.embedding_size, 
            embeddings_initializer=glorot_normal(seed=self.seed[i])) for i in range(self.num_lookup)]
        self.lookup_weights = self.add_weight(name='lookup_weight', shape=[self.num_lookup, 1], 
            initializer=glorot_uniform())

    def call(self, x):
        x = tf.expand_dims(x, axis=-1)
        embedding_list = [self.lookup_list[i](self.hash_fn_list[i](x)) for i in range(self.num_lookup)]
        embeddings = tf.concat(embedding_list, axis=-2)
        weighted_embedding = tf.reduce_sum(embeddings*self.lookup_weights, axis=1)
        norm_wemb = weighted_embedding / tf.reduce_sum(self.lookup_weights)

        return norm_wemb


class CompositionalEmbedding(Layer):
    '''
    Compositional Embeddings Using Complementary Partitions for Memory Efficient Recommendation Systems.
    '''
    def __init__(self):
        super().__init__()

    def build(self, input_shape):
        pass

    def call(self, x):
        pass


class DenseHashEmbedding(Layer):
    '''
    Learning to Embed Categorical Features without Embedding Tables for Recommendation
    '''
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
    he = HashEmbedding(32)
    print(he(string_input).shape)
