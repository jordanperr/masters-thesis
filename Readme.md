# Distillation before Verification of Deep Neural Networks
Jordan Perr-Sauer, CU Boulder, 2022

This repository contains exploratory, experimental, and visualization code for my thesis "Distillation before Verification of Deep Neural Networks"

## Important files for thesis

- Figures from XOR proof of concept: notebooks/thesis_figures/xor-example.ipynb
- Figures from ACAS-Xu Distillation: notebooks/thesis_figures/acasxu-distillation.ipynb
- Figures from ACAS-Xu Experiments: notebooks/thesis-figures/acasxu-experiment-visualization.ipynb
- ACAS-Xu Experiments that were run on Summit using the following code:
    - experiments/distill_before_verify_acasxu/0_run_full_experiment.sh
    - entrypoint is `0_run_full_experiment.sh some_config_file.json`, which takes as a parameter a json config file.
    - Config file for ndata experiment: experiments/distill_before_verify_acasxu/exp_summit_ndata_11_06_2022.json
    - Config file for depth experiment: experiments/distill_before_verify_acasxu/exp_summit_depth_11_06_2022.json

## Running on Summit

1. Check out this repository
2. Create distill-env anaconda environment.
3. Modify slurm_summit_run_experiment and the json config files to reflect your environment.
4. cd experiments/distill_before_verify_acasxu
5. sbatch --export=DISTILL_JOB_NAME=exp_summit_test_11_06_2022 --partition=shas-testing  --time=00:30:00 slurm_summit_run_experiment.sh

Actual experiments for thesis:

```
[jope8154@login12 distill_before_verify_acasxu]$ sbatch --export=DISTILL_JOB_NAME=exp_summit_depth_11_06_2022 --time=23:00:00 slurm_summit_run_experiment.sh
Submitted batch job 11931219
[jope8154@login12 distill_before_verify_acasxu]$ sbatch --export=DISTILL_JOB_NAME=exp_summit_ndata_11_06_2022 --time=23:00:00 slurm_summit_run_experiment.sh
Submitted batch job 11931238
```

## Other files in this repository:

- notebooks/*.ipynb: Mostly exploratory notebooks. Organized by date.

## Dependencies:
- ACAS-Xu Data and Properties: https://github.com/stanleybak/vnncomp2021/tree/79af0d7/benchmarks/acasxu
- NNenum Software: https://github.com/stanleybak/NNenum/tree/fa1463b