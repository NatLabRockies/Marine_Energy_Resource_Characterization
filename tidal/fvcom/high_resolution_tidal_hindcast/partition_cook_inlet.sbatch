#!/bin/bash
#SBATCH --account=hindcastra
#SBATCH --output=part_cook%A_%a.out
#SBATCH --job-name=part_cook
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=andrew.simms@nrel.gov
#SBATCH --time=24:00:00
##SBATCH --partition=standard
# Faces 392002
# Time Interval: Hourly
# Recommended Batch Size: 5000
# 5000 failed with OOM
# Reducing to 2500
# Calculated Tasks = 392002 / 5000 = 78.4004
# Calculated Tasks = 392002 / 2500 = 156.8
##SBATCH --array=0-78  # Create 79 tasks
#SBATCH --array=0-156  # Create 157 tasks
#SBATCH --mem=64G
#SBATCH --partition=shared

source sbatch_env_setup.sh

# Create a Python script that handles the parallelized processing
python partition_dataset.py cook_inlet --batch-size 2500 --batch-num $SLURM_ARRAY_TASK_ID 
