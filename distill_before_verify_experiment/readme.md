# Code to reproduce the experimental section in my Masters thesis

- 10_10_2022_experiment.py: Experimental code which takes networks and hyperparameters as input and produces output data.
- 10_10_2022_viz.py: Code to convert output data into visualizations and tables for the thesis.

## QuickStart

Requirements:

- Python 3.10
- POSIX-like operating system (Mac, Linux)
- NNenum

We suggest you create a new Conda environment or Virtual Environment:
```
conda create --name distill-env python=3.9
```

Install dependencies
```
pip install -r requirements.txt
```

Point to NNENUM installation
```
export PYTHONPATH=/Users/jperrsau/cu-src/thesis/src/nnenum/src:$PYTHONPATH
```

Write a configuration file like `test_experiment.json`

Run the experiment
```
bash 0_run_full_experiment.sh test_experiment
```

Inspect output in `./output`

## Docker Image

A pre-built docker image will be made available for download on github container registry.
