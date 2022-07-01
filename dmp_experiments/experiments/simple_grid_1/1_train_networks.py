## DMP Interpretability
## 1_train_networks.py
## Jordan Perr-Sauer, 2022
## Scan a simple grid of parameters using rectangle network, save out weights, model, and metrics to ./saved_models

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


class save_every_epoch_callback(tensorflow.keras.callbacks.Callback):

    def __init__(self, epochs, path):
        self.epochs=epochs
        self.path=path

    def on_epoch_end(self, epoch, logs=None):
        base_path = self.path+f"/epoch={epoch}"
        if epoch % self.epochs == 0:
            self.model.save(base_path)
            with open(base_path+'/logs.json', 'w') as outfile:
                json.dump(logs, outfile)


def run(depth=8, width=10, repetition=0, epoch_max=100, epoch_save_every=20):

    X_train, X_test, y_train, y_test = load_data()

    num_observations = X_train.shape[0]
    num_inputs = X_train.shape[1]
    num_outputs = y_train.shape[1]

    save_path = SAVE_DIR+"/"+f"experiment=simplegrid1/depth={depth}/width={width}/repetition={repetition}"

    model = build_network(depth,width,num_inputs,num_outputs)

    model.compile(
        # loss='binary_crossentropy', # binary classification
        # loss='categorical_crossentropy', # categorical classification (one hot)
        loss='mean_squared_error',  # regression
        optimizer=tensorflow.keras.optimizers.Adam(0.001),
        # optimizer='rmsprop',
        # metrics=['accuracy'],
    )

    model.fit(
        x=X_train,
        y=y_train,
        validation_data=(X_test, y_test),
        shuffle=True,
        epochs=100,
        batch_size=256,
        callbacks=[
            save_every_epoch_callback(20, save_path)
        ]
    )



if __name__=="__main__":
    for depth in [3,7,11]:
        for width in [3,7,11]:
            for repetition in range(2):
                run(depth, width, repetition)