import tensorflow
import json

from utils import load_data, NpEncoder
from network import build_network, on_every_n_epochs
import metrics

from pathlib import Path

SAVE_DIR="./saved_models"
PRNG_SEED=42

## Every 20 epochs save the expensive interp metrics
## Every 1 epochs save the loss metrics

class interpretability_metrics_callback(tensorflow.keras.callbacks.Callback):

    def __init__(self, epochs, path, data):
        self.epochs=epochs
        self.path=path
        self.data=data

    def on_epoch_end(self, epoch, logs=None):

        X_train, X_test, y_train, y_test = self.data

        base_path = self.path+f"/epoch={epoch}"
        if epoch % self.epochs == 0:

            out = {}
            
            out["sparsity"] = metrics.sparsity(self.model)
            out["basic_statistics"] = metrics.basic_statistics(self.model)
            out["layerwise_pca"] = metrics.layerwise_pca(self.model, X_test, y_test)
            out["linear_probes"] = metrics.linear_probes(self.model, X_test, y_test)
            out["cca"] = metrics.cca(self.model, X_test, y_test)
            out["logs"] = logs

            Path(base_path).mkdir(parents=True, exist_ok=True)
            with open(base_path+'/logs.json', 'w') as outfile:
                json.dump(out, outfile, cls=NpEncoder)


def run(depth=8, width=10, repetition=0, epoch_max=100, epoch_save_every=20):

    X_train, X_test, y_train, y_test = load_data()

    num_observations = X_train.shape[0]
    num_inputs = X_train.shape[1]
    num_outputs = y_train.shape[1]

    save_path = SAVE_DIR+"/"+f"experiment=simplegrid2/depth={depth}/width={width}/repetition={repetition}"

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
        epochs=101,
        batch_size=256,
        callbacks=[
            interpretability_metrics_callback(epoch_save_every, save_path, (X_train, X_test, y_train, y_test))
        ]
    )



if __name__=="__main__":
    for depth in [3,7,11]:
        for width in [3,7,11]:
            for repetition in range(2):
                run(depth, width, repetition)