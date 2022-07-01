from re import M
import tensorflow as tf
import numpy as np
from collections import defaultdict

from tensorflow.keras.wrappers.scikit_learn import KerasRegressor, KerasClassifier

from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error

import functools
import logging

import numpy

# Basic statistics of the weights and biases

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

def sparsity(model:tf.keras.Sequential):
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


# Network dissection using layerwise linear models

def _get_forward_activations(model:tf.keras.Sequential, X_data:numpy.ndarray):
    """
    Utility function to compute activations of each unit in model with respect to X_data
    """

    logging.info(f"Computing forward activations for model {model}")
    keras_inputs = [model.input]
    keras_outputs = [layer.output for layer in model.layers]
    activations = tf.keras.backend.function(keras_inputs, keras_outputs)([X_data])

    return activations

def layerwise_pca(model:tf.keras.Sequential, inputs, outputs, artifact_path=False):
    """
    Statistics related to the pca of data from each input.
    Reference: Technique inspired from CCA paper by Kornblith, but this is my own path from there.
    Inputs:
        - model (keras.Sequential): Keras model
        - inputs (np.array): Array of input data over which to compute the metrics
        - outputs (no.array): Array of outputs corresponding to the inputs
        - artifact_path (str|None): If str, path to directory to save image artifacts
    Output: Dict{str:Array(float)}
    """
    log = defaultdict(list)
    activations = _get_forward_activations(model, inputs)
    for activation in activations:
        pca = PCA()
        pca.fit(activation)
        log["explained_variance"].append(pca.explained_variance_)

    return log

def linear_probes(model:tf.keras.Sequential, inputs, outputs):
    """
    Compute accuracy of linear models trained on the output of each hidden layer.
    Reference: Alain and Bengio, 2018, Understanding intermediate layers using linear classifier probes
    Output: Array(float)
    """
    log = defaultdict(list)
    activations = _get_forward_activations(model, inputs)
    for activation in activations:
        lm_outputs = LinearRegression().fit(activation, outputs).predict(activation)
        log["mse"].append(mean_squared_error(outputs, lm_outputs))
        log["mae"].append(mean_absolute_error(outputs, lm_outputs))

    return log


## Layerwise correlation metrics

def linear_cka(model:tf.keras.Model, inputs, outputs):

    def cka(A,B):
        numerator = np.linalg.norm(A@B.T, ord="fro")**2
        denominator = np.linalg.norm(A@A.T, ord="fro")*np.linalg.norm(B@B.T, ord="fro")
        return 1-(numerator/denominator)

    activations = _get_forward_activations(model, inputs)

    layers = model.layers[:-1] # Last layer is the output layer, with only one neuron, so we must exclude it.

    result = np.ndarray((len(layers), len(layers)), dtype="float")

    for index_a, layer_a in enumerate(layers):
        for index_b, layer_b in enumerate(layers):
            result[index_a, index_b] = cka(activations[index_a],activations[index_b])

    return result

def cca(model:tf.keras.Model, inputs, outputs):

    activations = _get_forward_activations(model, inputs)
    ortho_activations = [np.linalg.qr(X)[0] for X in activations]

    layers = model.layers[:-1] # Last layer is the output layer, with only one neuron, so we must exclude it.

    result = np.ndarray((len(layers), len(layers)), dtype="float")

    for index_a, layer_a in enumerate(layers):
        for index_b, layer_b in enumerate(layers):

            qx = ortho_activations[index_a]
            qy = ortho_activations[index_b]
            features_x = activations[index_a]
            features_y = activations[index_b]

            cca_ab = np.linalg.norm(qx.T.dot(qy)) ** 2 / min(
                features_x.shape[1], features_y.shape[1])

            result[index_a, index_b] = cca_ab

    return result


## Global Sensitivity

def permutation_feature_importance(model:tf.keras.Model, inputs, outputs):
    pass

def partial_dependence_plots():
    pass

def prototypes_criticisms():
    pass

def ice_plots():
    pass

## Output / Class based metrics? CAVs?

## Output / Generative metrics?

## Compute All Metrics
def compute_all(model, inputs, outputs, artifact_path=None):
    pass