import tensorflow as tf
from tensorflow.python.keras.layers import Layer, Activation



try:
    unicode
except NameError:
    unicode = str

class Dice():
    def __init__(self) -> None:
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