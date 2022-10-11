# Distill / distill_before_verify_experiment / 10_10_2022_experiment.py
# Part of Scaling Verification with Knowledge Distillation
# Author: Jordan Perr-Sauer
# Date: 10/10/2022
# Copyright: All rights reserved

import time
import logging

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

import onnx
import onnx_tf
import onnxruntime

from pathlib import Path

logger = logging.getLogger()
rng = np.random.default_rng(seed=42)

class AcasXUNetwork:
    """
    Wrapper class that loads an ACASXu network from the VNNComp 2021 data archive using onnx and onnx_tf.
    Makes easier the interaction with the network, such as reshaping inputs.
    """
    def __init__(path):
        acas_xu = onnx.load(path)
        acas_xu = onnx_tf.backend.prepare(acas_xu)

    @classmethod
    def reshape_inputs(cls, inputs):
        n = inputs.shape[0]
        return inputs.reshape((n,1,1,5))

    def run(inputs):
        inputs = AcasXUNetwork.reshape_inputs(inputs)
        onnx_outputs = acas_xu.run(inputs)
        return onnx_outputs.linear_7_Add

#        inputs = (rng.random((n,1,1,5),dtype="float32")-0.5)*2

def load_vnncomp_2021_acasxu_network(tau, a_prev, path="../data/acasxu"):
    """
    Load an ACAS Xu neural network in onnx format from the VNNComp 2021 data archive.
    Katz, G., Barrett, C., Dill, D.L., Julian, K., Kochenderfer, M.J. (2017). Reluplex: An Efficient SMT Solver for Verifying Deep Neural Networks. In: Majumdar, R., Kunčak, V. (eds) Computer Aided Verification. CAV 2017. Lecture Notes in Computer Science(), vol 10426. Springer, Cham. https://doi.org/10.1007/978-3-319-63387-9_5
    Networks were downloaded from: https://github.com/stanleybak/vnncomp2021/tree/main/benchmarks/acasxu
    """
    return AcasXUNetwork(Path(path)/f"ACASXU_run2a_{tau}_{a_prev}_batch_2000.onnx")

def distill(teacher:AcasXUNetwork,
            n_synthetic_data_points,
            synthetic_data_sampling,
            hidden_layer_width,
            num_hidden_layers):
    """
    Data-free distillation (compression) of teacher network.

    Returns:
    - student_model(tf.model)
    """
    ## Generate synthetic input data for distillation process
    synthetic_inputs = (rng.random((n_synthetic_data_points,5),dtype="float32")-0.5)*2

    ## Run the teacher network on the synthetic input data
    synthetic_outputs = teacher.run(synthetic_inputs)
    
    ## Create student network
    ### Input Layer
    layers = [
        tf.keras.layers.Dense(
            hidden_layer_width,
            activation=tf.nn.relu,
            input_shape=(5,),
            kernel_initializer=tf.keras.initializers.GlorotUniform(),
            bias_initializer=tf.keras.initializers.GlorotUniform()
        )
    ]

    ### Hidden Layers
    for i in range(num_hidden_layers-1):
        layers.append(
            tf.keras.layers.Dense(hidden_layer_width,
                activation=tf.nn.relu,
                kernel_initializer=tf.keras.initializers.GlorotUniform(),
                bias_initializer=tf.keras.initializers.GlorotUniform())
        )

    ### Output Layer
    layers.append(tf.keras.layers.Dense(5))
    student_model = tf.keras.Sequential(layers)

    ### Julian K uses an asymmetric loss function based on MSE. We use MSE here for now.
    student_model.compile(
        loss=tf.keras.losses.MeanSquaredError(),
        #loss=tf.keras.losses.KLDivergence(),
        metrics=[tf.keras.metrics.MeanSquaredError()],
        optimizer=tf.keras.optimizers.Adam(0.001)
    )

    ## Fit the student model using the synthetic data
    student_model.fit(
        x=synthetic_inputs,
        y=synthetic_outputs,#synthetic_outputs, #- Using logits in the loss function requires a different loss metric, like KLDivergence. But I couldn't get it working immediately.
        epochs=500,
        batch_size=128,
        verbose=0,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor='loss', patience=20)
        ]
    )

    return student_model

def verify(network, property, verifier):
    """
    use an external verification tool to verify the network
    """
    pass

def distill_verify_comparison_experiment(network_path, property_path, student_options):
    teacher = load_network(network_path)
    
    # Classic Verification
    output = verify(student, property_path, timeout=600)

    # Distillation Verification
    student = distill(teacher, student_options)
    output = verify(student, property_path, timeout=600)

def refinement_loop_experiment():
    pass

if __name__=="__main__":
    PARALLELISM = 8

    hyperparameters = {
        "depths": [2,3,4,5],
        "widths": [2**n for n in range(2,9)],
        "repetitions": list(range(10)),
        "properties": ["p1", "p2", "p3", "p4"],
        "n_synthetic_data_points": [2**n for n in range(10,15)],
        "synthetic_data_sampling": ["qmc", "random_iid"],
        "network_tau": [1],
        "network_a_prev": [1],
    }
    #training_iterations = ["early_stopping", 2000]

    logger.info(f"Using {PARALLELISM} cores.")

    #for params in itertools.product(**hyperparameters.values())

        #acas_xu = distill_verify_comparison_experiment(tau=1, a_prev=1)