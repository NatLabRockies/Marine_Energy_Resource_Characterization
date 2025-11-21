#!/bin/bash
#SBATCH --account=hindcastra
#SBATCH --output=part_west_%A_%a.out
#SBATCH --job-name=part_west
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=andrew.simms@nrel.gov
#SBATCH --time=3:59:00
#SBATCH --partition=short
##SBATCH --mem=128G
# There are 231208 faces
# 2500 faces per batch
# 213208 / 2500 = 92.48 tasks
#SBATCH --array=0-92  # Create 93 tasks

source sbatch_env_setup.sh

# Create a Python script that handles the parallelized processing
python partition_dataset.py western_passage --batch-size 2500 --batch-num $SLURM_ARRAY_TASK_ID
