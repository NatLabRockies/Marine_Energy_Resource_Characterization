#!/bin/bash
#SBATCH --account=hindcastra
#SBATCH --output=part_puget_%A_%a.out
#SBATCH --job-name=part_puget
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=andrew.simms@nrel.gov
#SBATCH --time=24:00:00
#SBATCH --partition=standard
# There are 1734765 faces and batch size is 2500
# 1734765 / 10000 = 173.4765
# 1734765 / 5000 = 346.953
# 1734765 / 2500 = 693.906
##SBATCH --array=0-693  # Create 694 tasks
##SBATCH --array=0-173  # Create 174 tasks
#SBATCH --array=0-347  # Create 174 tasks
##SBATCH --mem=128G

source sbatch_env_setup.sh

# Create a Python script that handles the parallelized processing
python partition_dataset.py puget_sound --batch-size 5000 --batch-num $SLURM_ARRAY_TASK_ID 
