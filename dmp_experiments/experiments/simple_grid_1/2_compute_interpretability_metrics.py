## DMP Interpretability
## 2_compute_interpretability_metrics.py
## Jordan Perr-Sauer, 2022
## Compute interpretability metrics and add interpretability_metrics.json file to each experiment in saved_models

import glob
import pandas as pd
import json
import tensorflow
from utils import load_data, NpEncoder

import metrics

def run(path):
    model = tensorflow.keras.models.load_model(path)
    X_train, X_test, y_train, y_test = load_data() # might need to ensure test and train data is the same as it was at model generation...
    log = {}
    log["basic_statistics"] = metrics.basic_statistics(model)
    log["sparsity"] = metrics.sparcity(model)
    log["layerwise_pca"] = metrics.layerwise_pca(model, X_train, y_train)
    log["linear_probes"] = metrics.linear_probes(model, X_test, y_test)
    with open(path+"/interpretability_metrics.json", "w") as jsonfile:
        json.dump(log, jsonfile, cls=NpEncoder)
        print(path)

files = glob.glob("./saved_models/**/epoch=*/", recursive=True)
for path in files:
    run(path)

