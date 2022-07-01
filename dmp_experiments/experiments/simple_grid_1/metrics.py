from re import M
import tensorflow as tf
import numpy as np
from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Dataset agnostic metrics

def basic_statistics(model:tf.keras.Sequential):
    """
    Statistics related to basic statsitics on the distirbution of the value of the weights.
    Reference: ???
    Output: Dictionay[str,array[float]], where idx of array corresponds to layer number.
    """
    log = defaultdict(list)
    for layer in model.layers:
        weights, biases = layer.get_weights()
        log["weight_avg"].append(float(weights.mean()))
        log["bias_avg"].append(float(biases.mean()))
    return dict(log)

def sparcity(model:tf.keras.Sequential):
    """
    Statistics related to the sparsity of each layer.
    Reference: ???
    Output: Dictionay[str,array[float]], where idx of array corresponds to layer number.
    """
    log = defaultdict(list)
    for layer in model.layers:
        weights, biases = layer.get_weights()
        log["nnz_weights"].append( np.sum(np.abs(weights.flatten()) > 0.00001) )
        log["total_weights"].append( weights.flatten().shape[0] )
        log["nnz_biases"].append( np.sum(np.abs(biases.flatten()) > 0.00001) )
        log["total_biases"].append( biases.flatten().shape[0] )
    return dict(log)


# Metrics that require the dataset to compute

def layerwise_pca(model:tf.keras.Sequential, inputs, outputs):
    """
    Statistics related to the pca of data from each input.
    Reference: Alain and Bengio, 2018, Understanding intermediate layers using linear classifier probes
    Output: Dict{str:Array(float)}
    """
    log = defaultdict(list)

    for layer in model.layers:
        activations = tf.keras.backend.function([model.input], layer.output)([inputs])
        pca = PCA()
        pca.fit(activations)
        log["explained_variance"].append(pca.explained_variance_)

    return log

def linear_probes(model:tf.keras.Sequential, inputs, outputs):
    """
    Compute accuracy of linear models trained on the output of each hidden layer.
    Reference: Alain and Bengio, 2018, Understanding intermediate layers using linear classifier probes
    Output: Array(float)
    """
    log = defaultdict(list)

    for layer in model.layers:
        activations = tf.keras.backend.function([model.input], layer.output)([inputs])
        lm_outputs = LinearRegression().fit(activations, outputs).predict(activations)
        log["mse"].append(mean_squared_error(outputs, lm_outputs))
        log["mae"].append(mean_absolute_error(outputs, lm_outputs))

    return log


def cca(model:tf.keras.Model):
    pass