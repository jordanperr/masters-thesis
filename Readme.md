Thesis files
Jordan Perr-Sauer, CU Boulder, 2022

## Running on Summit

1. Check out this repository
2. Create distill-env anaconda environment.
3. Modify slurm_summit_run_experiment and the json config files to reflect your environment.
4. cd experiments/distill_before_verify_acasxu
5. sbatch --export=DISTILL_JOB_NAME=exp_summit_test_11_06_2022 --partition=shas-testing  --time=00:30:00slurm_summit_run_experiment.sh