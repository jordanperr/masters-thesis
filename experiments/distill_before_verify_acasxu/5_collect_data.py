import pandas as pd
import json
import re
import sys

print("5_collect_data.py")

with open(sys.argv[1]) as config_fp:
    global_config = json.load(config_fp)

path = global_config["output_dir"]

index = pd.read_csv(path+"/index.csv")

results = []

for row in index.itertuples():
    result = pd.read_csv(path+"/"+row.uuid+"/student.stats.csv")
    result["uuid"] = row.uuid

    for property in global_config["properties"]:
        with open(f"{path}/{row.uuid}/verify.{property}.result", "r") as fp:
            verified_result, python_time = [a.strip() for a in fp.readlines()]
            result[f"prop.{property}.result"] = verified_result.strip()
            result[f"prop.{property}.python_time"] = float(python_time.split(":")[1])
        
        with open(f"{path}/{row.uuid}/verify.{property}.stdout", "r") as fp:
            print(f"{path}/{row.uuid}/verify.{property}.stdout")
            contents = fp.read()
            print(f"{path}/{row.uuid}/verify.{property}.stdout")
            if re.search("Timeout \(\d+.\d+\) reached during execution", contents) != None:
                runtime_re = "TimeoutReached"
                result_re = "Unknown - Timeout Reached"
            elif re.search("Proven safe before enumerate", contents) != None:
                runtime_re = "SafeBeforeEnumerate"
                result_re = "Safe"
            else:
                runtime_re = re.search("Runtime: (\d+\.\d+)", contents).groups(0)[0]
                result_re = re.search("Result: ([a-zA-Z\s]+)", contents).groups(0)[0]

            result[f"prop.{property}.reported_runtime"] = runtime_re.strip()
            result[f"prop.{property}.stdout_result"] = result_re.strip()

    results.append(result)

results_df = pd.concat(results)

combined_results = pd.merge(index, results_df, on="uuid", how="left")

combined_results.to_csv(f"{path}/results.csv", index=False)

