#!/bin/bash
#SBATCH --account=hindcastra
#SBATCH --output=part_west_%A_%a.out
#SBATCH --job-name=part_west
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=andrew.simms@nrel.gov
#SBATCH --time=6:00:00
#SBATCH --partition=standard
##SBATCH --mem=128G
# There are 231208 faces
# 231208 / 5000 = 46.24 -> ceil = 47 tasks (0-46)
# 231208 / 2500 = 92.48 -> ceil = 93 tasks (0-92)
##SBATCH --array=0-92  # Create 93 tasks
#SBATCH --array=0-46  # Create 47 tasks (faces 0-231207)

source sbatch_env_setup.sh

# Create a Python script that handles the parallelized processing
python partition_dataset.py western_passage --batch-size 5000 --batch-num $SLURM_ARRAY_TASK_ID
