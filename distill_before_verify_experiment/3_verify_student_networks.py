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

path = sys.argv[1]

def run_verify(config):

    network_path = path+f"/{config['uuid']}/student.onnx"

    properties = ["1","2","3","4"]

    for property in properties:
        result_path = path+f"/{config['uuid']}/verify.{property}.result"
        stdout_path = path+f"/{config['uuid']}/verify.{property}.stdout"

        cmd = f"docker run -v /Users/jperrsau/cu-src/thesis/src/distill:/my_work nnenum_image bash -c \"python3 -m nnenum.nnenum /my_work/distill_before_verify_experiment/{network_path} /my_work/data/acasxu/prop_{property}.vnnlib 600 /my_work/distill_before_verify_experiment/{result_path} > /my_work/distill_before_verify_experiment/{stdout_path}\""
        #print(cmd)
        
        start = time.perf_counter()
        subprocess.getoutput(cmd)
        verify_time = time.perf_counter() - start

        ## Append perf_counter time to result file
        with open(result_path, "a") as f:
            f.write("\npython_time:"+str(verify_time))

if __name__=="__main__":
    experiments = pd.read_csv(path+"/index.csv").to_dict("records")

    with mp.Pool(4) as p:
        results = list(tqdm.tqdm(p.imap(run_verify, experiments), total=len(experiments)))