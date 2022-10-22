#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=24
#SBATCH --time=02:00:00
#SBATCH --account=ucb-general
#SBATCH --output=slurm_exp_alpine_ndata_val_10_17_2022-%j.out
#SBATCH --job-name distill-job

module purge
module load anaconda

cd /home/jope8154/projects/masters_thesis/distill/distill_before_verify_experiment

conda activate ./distill-env

export PYTHONPATH=/home/jope8154/projects/masters_thesis/src/nnenum/src:$PYTHONPATH
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

bash 0_run_full_experiment.sh exp_alpine_ndata_val_10_17_2022 

