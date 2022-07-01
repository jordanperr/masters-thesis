import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def print_layer_pca(model, inputs, outputs):
    layerwise_activations = []
    for layer in model.layers:
        keras_function = tf.keras.backend.function([model.input], [layer.output])
        layerwise_activations.append(keras_function([inputs]))

    plt.figure(figsize=(3*len(layerwise_activations)-1,3))
    plt.subplot(1,len(layerwise_activations),1)

    plt.tick_params(left=False,
                bottom=False,
                labelleft=False,
                labelbottom=False)

    pca = PCA(2)
    out = pca.fit_transform(inputs)
    plt.scatter(out[:,0], out[:,1], c=outputs)
    plt.title("Input Data PCA")

    for layer_idx, layer in enumerate(layerwise_activations[:-1]):
        plt.subplot(1,len(layerwise_activations),layer_idx+2)
        layer = layer[0]
        pca = PCA(2)
        out = pca.fit_transform(layer)
        plt.tick_params(left=False,
            bottom=False,
            labelleft=False,
            labelbottom=False)
        plt.scatter(out[:,0], out[:,1], c=outputs)
        plt.title(f"Layer {layer_idx} PCA")
    
    plt.show()