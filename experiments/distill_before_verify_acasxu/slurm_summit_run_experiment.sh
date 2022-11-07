#!/bin/bash
sbatch <<EOT
#!/bin/bash

#SBATCH --account=ucb-general
#SBATCH --partition=shas
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=24
#SBATCH --time=08:00:00
#SBATCH -o="slurm_out."$1".txt"
#SBATCH -e="slurm_err."$1".txt"
#SBATCH --job-name distill-$1

module purge
module load anaconda
conda activate /home/jope8154/projects/masters_thesis/distill-env

lscpu > exp_alpine_depth_val_10_17_2022/lscpu.out
conda list --export > exp_alpine_depth_val_10_17_2022/requirements-freeze.txt

cd /home/jope8154/projects/masters_thesis/distill/experiments/distill_before_verify_experiment

export PYTHONPATH=/home/jope8154/projects/masters_thesis/src/nnenum/src:$PYTHONPATH
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

bash 0_run_full_experiment.sh $1

exit 0
EOT