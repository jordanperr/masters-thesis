"""
Step 2 in the distill_before_verify pipeline

Inputs:
- index.csv

Outputs:
- UUID.student.onnx
- UUID.student.history.csv
- UUID.student.stats.csv
"""

from experiment_lib import load_vnncomp_2021_acasxu_network, distill, check_closeness, write_tf_network_to_onnx_file
import pandas as pd
import sys
import multiprocessing as mp
import tqdm
import time
import pathlib
import json
from keras.utils.layer_utils import count_params

path = sys.argv[1]

with open(sys.argv[1]+".json") as config_fp:
    global_config = json.load(config_fp)

def run_distill(params):

    pathlib.Path(path+f"/{params['uuid']}").mkdir(parents=True, exist_ok=False)

    # Read teacher network from disk or cache
    teacher = load_vnncomp_2021_acasxu_network(params["tau"], params["a_prev"])

    # Create student network using network distillation
    start = time.perf_counter()

    student, history = distill(teacher,
        params["n_synthetic_data_points"],
        params["synthetic_data_sampling"],
        params["hidden_layer_width"],
        params["num_hidden_layers"])

    distill_time = time.perf_counter() - start

    # Write student network to disk
    write_tf_network_to_onnx_file(student, f"{path}/{params['uuid']}/student.onnx")

    # Write stats about student network training to disk
    history = pd.DataFrame(history.history)
    history.to_csv(f"{path}/{params['uuid']}/student.history.csv", index=False)

    stats = history.iloc[-1:].copy()
    stats["distill_time"] = distill_time
    stats["trainable_parameters"] = count_params(student.trainable_weights)

    stats.to_csv(f"{path}/{params['uuid']}/student.stats.csv", index=False)


### Main Loop
if __name__=="__main__":
    print("2_generate_student_networks.py")

    experiments = pd.read_csv(path+"/index.csv").to_dict("records")

    with mp.Pool(global_config["distill_parallelism"]) as p:
        results = list(tqdm.tqdm(p.imap(run_distill, experiments), total=len(experiments)))