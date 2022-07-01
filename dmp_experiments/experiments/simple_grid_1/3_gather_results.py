## DMP Interpretability
## 3_gather_results.py
## Jordan Perr-Sauer, 2022
## Recurse through the saved_models folder and generate csv file and any other aggregate data that will be helpful for analysis

import glob
import pandas as pd
import json

files = glob.glob("./saved_models/**/epoch=*/", recursive=True)

def get_records(files):
    for path in files:
        attrs = {key:val for key,val in [ x.split("=") for x in path.split("/")[2:-1] ]}
        with open(path+"/logs.json", "r") as jsonfile:
            attrs.update(json.load(jsonfile))
        with open(path+"/interpretability_metrics.json", "r") as jsonfile:
            attrs.update(json.load(jsonfile))
        yield attrs

df = pd.DataFrame.from_dict(get_records(files))

df.to_csv("results.csv", index=False)