#!/bin/bash
#SBATCH --account=hindcastra
#SBATCH --output=part_aleutian_%A_%a.out
#SBATCH --job-name=part_cook
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=andrew.simms@nrel.gov
#SBATCH --time=24:00:00
##SBATCH --partition=standard
# Faces 392002
# Time Interval: Hourly
# Recommended Batch Size: 2500
# Calculated Tasks = 797987 / 2500 = 319.19
#SBATCH --array=0-319  # Create 320 tasks
#SBATCH --mem=64G
#SBATCH --partition=shared

source sbatch_env_setup.sh

# Create a Python script that handles the parallelized processing
python partition_dataset.py aleutian_islands --batch-size 2500 --batch-num $SLURM_ARRAY_TASK_ID 
