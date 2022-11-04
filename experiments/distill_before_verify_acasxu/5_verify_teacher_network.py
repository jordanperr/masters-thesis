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

name = sys.argv[1]

with open(name+".json", "r") as fp:
    config = json.load(fp)

def run_verify(params):

    pathlib.Path(name+f"/teacher/{params['uuid']}").mkdir(parents=True, exist_ok=True)

    network_path = str(pathlib.Path("../../data")/f"acasxu/ACASXU_run2a_{params['tau']}_{params['a_prev']}_batch_2000.onnx")

    for property in config["properties"]:
        result_path = name+f"/teacher/{params['uuid']}/verify.{property}.result"
        stdout_path = name+f"/teacher/{params['uuid']}/verify.{property}.stdout"

        cmd = config["nnenum_cmd"].format(network_path=network_path,
            stdout_path=stdout_path,
            property=property,
            result_path=result_path)
        
        print(cmd)
        start = time.perf_counter()
        subprocess.getoutput(cmd)
        verify_time = time.perf_counter() - start

        ## Append perf_counter time to result file
        with open(result_path, "a") as f:
            f.write("\npython_time:"+str(verify_time))


if __name__=="__main__":

    experiments = pd.read_csv(name+"/index.csv", usecols=["repetition","tau","a_prev"]).drop_duplicates()
    experiments["uuid"] = experiments.apply(lambda _: uuid.uuid4(), axis=1)
    experiments = experiments.to_dict("records")

    with mp.Pool(config["nnenum_parallelism"]) as p:
        results = list(tqdm.tqdm(p.imap(run_verify, experiments), total=len(experiments)))
