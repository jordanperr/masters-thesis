
module purge
module load anaconda

cd /home/jope8154/projects/masters_thesis/distill/distill_before_verify_experiment

conda activate ./distill-env

export PYTHONPATH=/home/jope8154/projects/masters_thesis/src/nnenum/src:$PYTHONPATH

bash 0_run_full_experiment.sh exp_alpine_depth_val_10_17_2022 

