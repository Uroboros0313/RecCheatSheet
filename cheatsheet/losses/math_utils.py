import tensorflow as tf


def tf_log2(x):
    log2_ = tf.divide(tf.math.log(x), tf.math.log(2.0))
    return log2_
