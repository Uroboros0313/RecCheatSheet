import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.initializers import glorot_uniform


class UncertainlyWeighLoss(Layer):
    def __init__(self, num_task):
        self.num_task = num_task
        
    def build(self, input_shape):
        self.uw_k = self.add_weight(name='UW_DynamicWeight', 
                                    shape=[self.num_task, 1], 
                                    initializer=glorot_uniform)
        
    def call(self, losses):
        """
        param losses: input shape [Batch_size, num_task]
        """
        
        uw_loss = tf.matmul(losses, 1.0 / tf.math.exp(self.uw_k)) + self.uw_k
        loss_ = tf.reduce_sum(uw_loss)
        
        return loss_