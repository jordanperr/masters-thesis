from pprint import pprint

import pandas
import tensorflow
from matplotlib import pyplot

from dmp.data.pmlb import pmlb_loader

import os
import sys

import numpy as np

'''
PMLB Homepage: https://github.com/EpistasisLab/pmlb
Python Reference: https://epistasislab.github.io/pmlb/python-ref.html



"537_houses" regression, 20640 observations, 8 features, 1 output
"529_pollen" regression, 3848 observations, 4 features, 1 output
"560_bodyfat" regression, 252 observations, 14 variables, 1 output


"adult" classification, 48842 observations, 14 features, 2 classes
"nursery" classification, 12958 observations, 8 features, 7 classes
"ring" classification, 7400 observations, 19 features, 2 classes
"satimage" classification, 6435 observations, 19 features, 6 classes
"cars"  classification, 392 observations, 8 features, 3 classes
"wine_recognition" classification, 178 observations, 13 features, 3 classes
"titanic" classification, 2201 observations, 3 features, 2 classes

4544_GeographicalOriginalofMusic	1059	117		continuous	0	regression

'''
def get_basic_537_houses(hidden_layer_width=128, num_hidden_layers=4, layer_array=None, num_epochs=100, seed=1, cache_dir=False, train_model=True):
    datasets = pmlb_loader.load_dataset_index()
    dataset, inputs, outputs = pmlb_loader.load_dataset(datasets, '537_houses')
    return get_basic(hidden_layer_width, num_hidden_layers, layer_array, num_epochs, seed, cache_dir, train_model, inputs, outputs, type="basic_537_houses")


def get_basic(hidden_layer_width=128, num_hidden_layers=4, layer_array=None, num_epochs=100, seed=1, cache_dir=False, train_model=True, inputs=None, outputs=None, type="basic"):

    run_name = f"type={type}-hidden_layer_width={hidden_layer_width}-num_hidden_layers={num_hidden_layers}-seed={seed}"

    tensorflow.random.set_seed(seed)

    pprint(inputs.shape)
    pprint(outputs)
    print(outputs.shape)

    num_observations = inputs.shape[0]
    num_inputs = inputs.shape[1]
    num_outputs = outputs.shape[1]

    print('num_observations {} num_inputs {} num_outputs {}'.format(num_observations, num_inputs, num_outputs))

    ## Build model or load it from cache

    if cache_dir:
        cache_directory = os.path.join(os.getcwd(), cache_dir, run_name)
        os.makedirs(cache_directory, exist_ok=True)
        try:
            model = tensorflow.keras.models.load_model(cache_directory)
            print(f"Cache hit, loading {run_name} from cache")
            return model, inputs, outputs
        except:
            print(f"Cache miss, running {run_name}")
            pass

    if layer_array == None:
        layers = [ tensorflow.keras.layers.Dense(hidden_layer_width, activation=tensorflow.nn.relu, input_shape=(num_inputs,))]
        for i in range(num_hidden_layers-1):
            layers.append( tensorflow.keras.layers.Dense(hidden_layer_width, activation=tensorflow.nn.relu))
        layers.append(tensorflow.keras.layers.Dense(num_outputs, activation=tensorflow.nn.sigmoid))
    else:
        layers = layer_array

    model = tensorflow.keras.Sequential(layers)

    if train_model:
        model.compile(
        # loss='binary_crossentropy', # binary classification
        # loss='categorical_crossentropy', # categorical classification (one hot)
        loss='mean_squared_error',  # regression
        optimizer=tensorflow.keras.optimizers.Adam(0.001),
        # optimizer='rmsprop',
        # metrics=['accuracy'],
        )

        model.fit(
            x=inputs,
            y=outputs,
            shuffle=True,
            validation_split=.2,
            # epochs=12,
            epochs=100,
            batch_size=256,
            )

    if cache_dir:
        model.save(cache_directory)

    return model, inputs, outputs


def get_basic_xor(hidden_layer_width=128, num_hidden_layers=4, num_epochs=100, seed=1, cache_dir=False, train_model=True):
    centroids = [(0,0,0), (0,1,1), (1,0,1), (1,1,0)]
    covariance = np.diag([0.5, 0.5])
    n_data_points = 2000

    def get_clusters():
        for centroid in centroids:
            x,y,z = centroid
            a = np.empty(shape=(int(n_data_points/4),3), dtype="float")
            a[:,0:2] = np.random.multivariate_normal([x,y], covariance, size=(int(n_data_points/4)))
            a[:,2] = z
            yield a

    data = np.concatenate(list(get_clusters()))
    inputs = data[:,0:2]
    outputs = data[:,2:3]

    return get_basic(hidden_layer_width, num_hidden_layers, seed=seed, cache_dir=cache_dir, inputs=inputs, outputs=outputs, type="basic_xor")

