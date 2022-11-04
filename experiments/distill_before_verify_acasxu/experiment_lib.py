# Distill / distill_before_verify_experiment / 10_10_2022_experiment.py
# Part of Scaling Verification with Knowledge Distillation
# Author: Jordan Perr-Sauer
# Date: 10/10/2022
# Copyright: All rights reserved

import time
import logging
import sys
import subprocess
import re
import json

import numpy as np
import pandas as pd
import tensorflow as tf

import onnx
import onnx_tf
import onnxruntime
import tf2onnx

from pathlib import Path

logger = logging.getLogger()
rng = np.random.default_rng(seed=42)

class AcasXUNetwork:
    """
    Wrapper class that loads an ACASXu network from the VNNComp 2021 data archive using onnx and onnx_tf.
    Makes easier the interaction with the network, such as reshaping inputs.
    """
    def __init__(self, path):
        self.acas_xu = onnx.load(path)
        self.acas_xu = onnx_tf.backend.prepare(self.acas_xu)

    @classmethod
    def reshape_inputs(cls, inputs):
        n = inputs.shape[0]
        return inputs.reshape((n,1,1,5))

    def run(self, inputs):
        inputs = AcasXUNetwork.reshape_inputs(inputs)
        onnx_outputs = self.acas_xu.run(inputs)
        return onnx_outputs.linear_7_Add

#        inputs = (rng.random((n,1,1,5),dtype="float32")-0.5)*2

def load_vnncomp_2021_acasxu_network(tau, a_prev, path="../../data/acasxu"):
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
    synthetic_inputs = (rng.random((n_synthetic_data_points,5),dtype="float32")-0.5)
    ## Run the teacher network on the synthetic input data
    synthetic_outputs = teacher.run(synthetic_inputs)

    ## Generate synthetic validation data for distillation process
    synthetic_inputs_val = (rng.random((int(n_synthetic_data_points*0.2),5),dtype="float32")-0.5)
    ## Run the teacher network on the synthetic input data
    synthetic_outputs_val = teacher.run(synthetic_inputs_val)
    
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
        #loss=tf.keras.metrics.SparseCategoricalCrossentropy(),
        loss=tf.keras.losses.MeanSquaredError(),
        #loss=tf.keras.losses.KLDivergence(),
        metrics=[tf.keras.metrics.MeanSquaredError(),
                 tf.keras.metrics.KLDivergence(),
                 tf.keras.metrics.CategoricalCrossentropy(),
                 tf.keras.metrics.CategoricalAccuracy()],
        optimizer=tf.keras.optimizers.Adam(0.001)
    )

    ## Fit the student model using the synthetic data
    history = student_model.fit(
        x=synthetic_inputs,
        y=synthetic_outputs,#synthetic_outputs, #- Using logits in the loss function requires a different loss metric, like KLDivergence. But I couldn't get it working immediately.
        epochs=500,
        batch_size=128,
        verbose=0,
        validation_data = (synthetic_inputs_val, synthetic_outputs_val),
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)
        ],
    )

    return student_model, history

def check_closeness(student, teacher):
    synthetic_inputs = (rng.random((2000,5),dtype="float32")-0.5)*2
    teacher_outputs = teacher.run(synthetic_inputs)
    student_outputs = student.predict(synthetic_inputs)
    mse = np.sqrt(np.sum(
        np.sqrt(np.sum((teacher_outputs - student_outputs)**2, axis=1))**2
        ))
    return mse

def write_tf_network_to_onnx_file(network, path):
    #input_signature = [tf.TensorSpec([5], tf.float32, name='x')]
    tf2onnx.convert.from_keras(network, output_path=path)

def verify(network, property, timeout=600):
    """
    use an external verification tool to verify an ONNX network with respect to a property

    inputs:
    - network (str): Path to network in ONNX file format.
    - property (str): Number of property to verify

    returns:
    - Runtime (Float)
    - Result (String)
    
    """
    cmd = f"docker run -v /Users/jperrsau/cu-src/thesis/src/distill:/my_work nnenum_image python3 -m nnenum.nnenum /my_work/distill_before_verify_experiment/{network} /my_work/data/acasxu/prop_{property}.vnnlib"
    print(cmd)
    result = subprocess.getoutput(cmd)
    print(result)

    if re.search("Proven safe before enumerate", result) != None:
        runtime_re = 0.0
        result_re = "Safe"
    else:
        runtime_re = re.search("Runtime: (\d+\.\d+)", result).groups(0)[0]
        result_re = re.search("Result: ([a-zA-Z\s]+)", result).groups(0)[0]

    return float(runtime_re), result_re

def distill_verify_comparison_experiment(
        tau, 
        a_prev,
        n_synthetic_data_points,
        synthetic_data_sampling,
        hidden_layer_width,
        num_hidden_layers,
        properties,
        repetition):
    
    # Read teacher network from disk or cache
    teacher = load_vnncomp_2021_acasxu_network(tau, a_prev)

    # Create student network using network distillation
    start = time.perf_counter()
    student = distill(teacher,
        n_synthetic_data_points,
        synthetic_data_sampling,
        hidden_layer_width,
        num_hidden_layers)

    distill_time = time.perf_counter() - start
    distill_mse = check_closeness(student, teacher)

    output = {
        "distill_time": distill_time,
        "distill_mse": float(distill_mse),
    }

    # Write student network to disk
    write_tf_network_to_onnx_file(student, "./tmp_onnx_network")

    for property in properties:
        verify_time, verify_result = verify("./tmp_onnx_network", property)
        output[f"verify_time_{property}"] = verify_time
        output[f"verify_result_{property}"] = verify_result

    return output

def run_experiment(params):
    return distill_verify_comparison_experiment(**params)

if __name__=="__main__":
    import multiprocessing as mp
    import tqdm
    import itertools

    PARALLELISM = 8

    #print( distill_verify_comparison_experiment(1,1,2000,"random_iid", 4,4,["1"]) )

    # hyperparameters = {
    #     "num_hidden_layers": [2,3,4,5],
    #     "hidden_layer_width": [2**n for n in range(2,9)],
    #     "repetition": list(range(10)),
    #     "properties": [["p1", "p2", "p3", "p4"]],
    #     "n_synthetic_data_points": [2**n for n in range(10,15)],
    #     "synthetic_data_sampling": ["random_iid"],
    #     "tau": [1],
    #     "a_prev": [1]
    # }

    hyperparameters = {
        "num_hidden_layers": [2,5],
        "hidden_layer_width": [2**n for n in [8]],
        "repetition": [1],
        "properties": [["1", "2", "3", "4"]],
        "n_synthetic_data_points": [2**n for n in [10,14]],
        "synthetic_data_sampling": ["random_iid"],
        "tau": [1],
        "a_prev": [1]
    }

    logger.info(f"Using {PARALLELISM} cores.")

    keys = hyperparameters.keys()
    vals = list(hyperparameters.values())
    items = list(itertools.product(*vals))
    items = [dict(zip(keys, item)) for item in items]
    with mp.Pool(PARALLELISM) as p:
        results = list(tqdm.tqdm(p.imap(run_experiment, items), total=len(items)))

    with open("result.json", "w") as result_file:
        json.dump(results, result_file)