#!/bin/bash
#SBATCH --account=hindcastra
#SBATCH --output=sum_cook_process_%A_%a.out
#SBATCH --job-name=sum_cook_process
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=andrew.simms@nrel.gov
#SBATCH --time=24:00:00
#SBATCH --partition=standard
#SBATCH --array=0-39  # Create 40 tasks

# Faces 392002
# Time Interval: Hourly
# Recommended Batch Size: 10000
# Calculated Tasks = 392002 / 10000 = 39.2

# Source shared environment setup
source sum_cook_inlet_setup.sh

echo "Starting processing task $SLURM_ARRAY_TASK_ID..."

# Create a Python script that handles the parallelized processing
python summarize_dataset.py cook_inlet --batch-size 10000 --batch-num $SLURM_ARRAY_TASK_ID 

echo "Task $SLURM_ARRAY_TASK_ID completed successfully."
