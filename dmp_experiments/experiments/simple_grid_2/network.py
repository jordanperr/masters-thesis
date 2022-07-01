
import tensorflow
import json
from utils import load_data

SAVE_DIR="./saved_models"
PRNG_SEED=42

def build_network(depth=2,width=2,num_inputs=1,num_outputs=1):
    layers = [ tensorflow.keras.layers.Dense(width, activation=tensorflow.nn.relu, input_shape=(num_inputs,))]
    for __ in range(depth-1):
        layers.append( tensorflow.keras.layers.Dense(width, activation=tensorflow.nn.relu))
    layers.append(tensorflow.keras.layers.Dense(num_outputs, activation=tensorflow.nn.sigmoid))
    model = tensorflow.keras.Sequential(layers)
    return model


class on_every_n_epochs(tensorflow.keras.callbacks.Callback):

    def __init__(self, n, do_this):
        self.epochs=n
        self.do_this=do_this

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.epochs == 0:
            self.do_this(self.model)