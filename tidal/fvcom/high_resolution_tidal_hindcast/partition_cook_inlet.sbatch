#!/bin/bash
#SBATCH --account=hindcastra
#SBATCH --output=part_cook%A_%a.out
#SBATCH --job-name=part_cook
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=andrew.simms@nrel.gov
#SBATCH --time=12:00:00
# Faces 392002
# Time Interval: Hourly
# Recommended Batch Size: 5000
# 5000 failed with OOM
# Reducing to 2500
# Calculated Tasks = 392002 / 10000 = 39.20
# Calculated Tasks = 392002 / 5000 = 78.4004
# Calculated Tasks = 392002 / 2500 = 156.8
# Calculated Tasks = 392002 / 1000 = 393
#SBATCH --array=0-38  # Create 39 tasks
##SBATCH --array=0-78  # Create 79 tasks
##SBATCH --array=0-156  # Create 157 tasks
##SBATCH --array=0-392  # Create 393 tasks
##SBATCH --mem=128G
#SBATCH --partition=standard

source sbatch_env_setup.sh

# Create a Python script that handles the parallelized processing
python partition_dataset.py cook_inlet --batch-size 10000 --batch-num $SLURM_ARRAY_TASK_ID 
