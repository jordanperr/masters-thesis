"""
Step 3 in the distill_before_verify pipeline

Inputs:
- index.csv
- UUID.student.onnx

Outputs:
- UUID.student.verify.PROPERTY.result
- UUID.student.verify.PROPERTY.stdout
"""
import sys
import pandas as pd
import multiprocessing as mp
import tqdm
import subprocess
import time
import json
import uuid
import pathlib

with open(sys.argv[1]) as config_fp:
    global_config = json.load(config_fp)

path = global_config["output_dir"]

def run_verify(config):

    pathlib.Path(path+f"/teacher/{config['uuid']}").mkdir(parents=True, exist_ok=True)

    network_path = global_config["acasxu_dir"] + f"/ACASXU_run2a_{config['tau']}_{config['a_prev']}_batch_2000.onnx"

    for property in global_config["properties"]:
        result_path = path+f"/teacher/{config['uuid']}/verify.{property}.result"
        stdout_path = path+f"/teacher/{config['uuid']}/verify.{property}.stdout"
        property_path = global_config["acasxu_dir"] + f"/prop_{property}.vnnlib"

        cmd = global_config["nnenum_cmd"].format(
            network_path=network_path,
            stdout_path=stdout_path,
            property_path=property_path,
            timeout=global_config["nnenum_timeout"],
            result_path=result_path)
        
        print(cmd)
        start = time.perf_counter()
        subprocess.getoutput(cmd)
        verify_time = time.perf_counter() - start

        ## Append perf_counter time to result file
        with open(result_path, "a") as f:
            f.write("\npython_time:"+str(verify_time))


if __name__=="__main__":
    print("4_verify_teacher_network.py")

    experiments = pd.read_csv(path+"/index.csv", usecols=["repetition","tau","a_prev"]).drop_duplicates()
    experiments["uuid"] = experiments.apply(lambda _: uuid.uuid4(), axis=1)
    experiments = experiments.to_dict("records")

    with mp.Pool(global_config["nnenum_parallelism"]) as p:
        results = list(tqdm.tqdm(p.imap(run_verify, experiments), total=len(experiments)))
