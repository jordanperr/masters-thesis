import pandas as pd
import json
import re
import sys
import glob
import os

print("6_collect_teacher_data.py")

name=sys.argv[1]

with open(name+".json", "r") as fp:
    config = json.load(fp)

uuids = os.listdir(name+"/teacher")
print(uuids)

index = pd.read_csv(name+"/index.csv")

results = []

for uuid in uuids:

    result={}

    print(uuid)

    for property in config["properties"]:

        with open(f"{name}/teacher/{uuid}/verify.{property}.result", "r") as fp:
            verified_result, python_time = [a.strip() for a in fp.readlines()]
            result[f"prop.{property}.result"] = verified_result.strip()
            result[f"prop.{property}.python_time"] = float(python_time.split(":")[1])
        
        if verified_result.strip() == "error":
            runtime_re = None
            result_re = None   
        else:
            with open(f"{name}/teacher/{uuid}/verify.{property}.stdout", "r") as fp:
                contents = fp.read()
                print(f"{name}/teacher/{uuid}/verify.{property}.stdout")
                if re.search("Timeout \(\d+.\d+\) reached during execution", contents) != None:
                    runtime_re = None
                    result_re = "Unknown - Timeout Reached"
                elif re.search("Proven safe before enumerate", contents) != None:
                    runtime_re = "0.0"
                    result_re = "Safe"
                else:
                    runtime_re = re.search("Runtime: (\d+\.\d+)", contents).groups(0)[0]
                    result_re = re.search("Result: ([a-zA-Z\s]+)", contents).groups(0)[0]

                result[f"prop.{property}.reported_runtime"] = runtime_re.strip()
                result[f"prop.{property}.stdout_result"] = result_re.strip()
    print(result)
    results.append(result)

results_df = pd.DataFrame(results)

results_df.to_csv(f"{name}/results.teacher.csv", index=False)

