#!/bin/bash

#SBATCH --account=ucb-general
#SBATCH --partition=shas
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=24
#SBATCH --time=08:00:00
#SBATCH --job-name distill-job

module purge
module load anaconda
conda activate /home/jope8154/projects/masters_thesis/distill-env

cd /projects/jope8154/masters_thesis/distill/experiments/distill_before_verify_acasxu

lscpu > slurm-$SLURM_JOB_ID.$DISTILL_JOB_NAME.lscpu.txt
conda list --export > slurm-$SLURM_JOB_ID.$DISTILL_JOB_NAME.conda-requirements-frozen.txt

export PYTHONPATH=/home/jope8154/projects/masters_thesis/src/nnenum/src:$PYTHONPATH
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=-1

bash 0_run_full_experiment.sh $DISTILL_JOB_NAME.json

touch slurm-$SLURM_JOB_ID.$DISTILL_JOB_NAME.isdone

exit 0