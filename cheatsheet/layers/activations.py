import tensorflow as tf
from tensorflow.keras.layers import Activation, Layer

try:
    unicode
except NameError:
    unicode = str


class Dice(Layer):
    def __init__(self) -> None:
        super().__init__()

    def build(self, input_shape):
        pass

    def call(self):
        pass


def build_act_fn(activation='relu'):
    if activation in ("dice", "Dice"):
        act_fn = Dice()
    elif isinstance(activation, (str, unicode)):
        act_fn = Activation(activation)
    elif issubclass(activation, Layer):
        act_fn = activation()
    else:
        raise ValueError(
            "Invalid activation,found %s.You should use a str or a Activation Layer Class." % (activation))
    return act_fn
