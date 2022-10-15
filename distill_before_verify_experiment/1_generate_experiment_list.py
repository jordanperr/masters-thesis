import itertools
import json
import sys
import pandas as pd
import pathlib
import uuid

#    "properties": [["1", "2", "3", "4"]],

# hyperparameters = {
#     "num_hidden_layers": [2,3,4,5],
#     "hidden_layer_width": [2**n for n in range(2,10)],
#     "repetition": list(range(10)),
#     "n_synthetic_data_points": [2**n for n in range(8,12)],
#     "synthetic_data_sampling": ["random_iid"],
#     "tau": [1],
#     "a_prev": [1]
# }

with open(sys.argv[1]+".json") as config_fp:
    config = json.load(config_fp)

keys = config["hyperparameters"].keys()
vals = list(config["hyperparameters"].values())

items = list(itertools.product(*vals))
items = [dict(zip(keys, item)) for item in items]
items = pd.DataFrame(items)
items['uuid'] = items.apply(lambda _: uuid.uuid4(), axis=1)

pathlib.Path(sys.argv[1]).mkdir(parents=True, exist_ok=False)
items.to_csv(sys.argv[1]+"/index.csv", index=False)

print(f"Generated {len(items)} experiments")