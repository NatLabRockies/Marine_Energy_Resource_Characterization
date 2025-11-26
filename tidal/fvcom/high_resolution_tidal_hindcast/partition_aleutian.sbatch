#!/bin/bash
#SBATCH --account=hindcastra
#SBATCH --output=part_aleutian_%A_%a.out
#SBATCH --job-name=part_aleutian
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=andrew.simms@nrel.gov
#SBATCH --time=6:00:00
#SBATCH --partition=standard
# Faces 797978 (from config.py)
# Time Interval: Hourly
# 797978 / 10000 = 79.80 -> ceil = 80 tasks (0-79)
# 797978 / 5000 = 159.60 -> ceil = 160 tasks (0-159)
# 797978 / 2500 = 319.19 -> ceil = 320 tasks (0-319)
# 797978 / 1000 = 797.98 -> ceil = 798 tasks (0-797)
#SBATCH --array=0-79  # Create 80 tasks
##SBATCH --array=0-159  # Create 160 tasks
##SBATCH --array=0-319  # Create 320 tasks
##SBATCH --array=0-797
##SBATCH --mem=64G
##SBATCH --mem=128G

source sbatch_env_setup.sh

# Create a Python script that handles the parallelized processing
python partition_dataset.py aleutian_islands --batch-size 10000 --batch-num $SLURM_ARRAY_TASK_ID
