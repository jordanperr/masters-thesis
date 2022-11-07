import itertools
import json
import sys
import pandas as pd
import pathlib
import uuid
import platform
import datetime
import subprocess
import random

with open(sys.argv[1]) as config_fp:
    config = json.load(config_fp)

keys = config["hyperparameters"].keys()
vals = list(config["hyperparameters"].values())

items = list(itertools.product(*vals))
items = [dict(zip(keys, item)) for item in items]
items = pd.DataFrame(items)
items['uuid'] = items.apply(lambda _: uuid.uuid4(), axis=1)
items['seed'] = items.apply(lambda _: random.randint(0,2**16), axis=1)

pathlib.Path(config["output_dir"]).mkdir(parents=True, exist_ok=False)
items.to_csv(config["output_dir"]+"/index.csv", index=False)

config["run_info"] = {
    "hostname": str(platform.node()),
    "timestamp": str(datetime.datetime.now().timestamp()),
    "distill_git_hash": str(subprocess.check_output(["git", "log", "-1", "--format=\"%H\""]))
}

with open(config["output_dir"]+"/config.json", "w") as fp:
    json.dump(config, fp)

print(f"Generated {len(items)} experiments")