#!/bin/bash
#SBATCH --account=hindcastra
#SBATCH --output=part_west_%A_%a.out
#SBATCH --job-name=part_west
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=andrew.simms@nrel.gov
#SBATCH --time=8:00:00
#SBATCH --partition=shared
#SBATCH --mem=64G
# There are 231208 faces and batch size is 2500
#SBATCH --array=0-92  # Create 93 tasks

source sbatch_env_setup.sh

# Create a Python script that handles the parallelized processing
python partition_dataset.py western_passage --batch-size 2500 --batch-num $SLURM_ARRAY_TASK_ID 
