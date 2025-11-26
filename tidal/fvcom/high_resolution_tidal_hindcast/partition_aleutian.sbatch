#!/bin/bash
#SBATCH --account=hindcastra
#SBATCH --output=part_aleutian_%A_%a.out
#SBATCH --job-name=part_aleutian
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=andrew.simms@nrel.gov
#SBATCH --time=18:00:00
#SBATCH --time=6:00:00
##SBATCH --time=3:59:59
#SBATCH --partition=standard
# Faces 797987
# Time Interval: Hourly
# Recommended Batch Size: 2500
# Recommended Batch Size: 1000
# Calculated Tasks = 797987 / 10000 = 79.7987
# Calculated Tasks = 797987 / 5000 = 159.5974
# Calculated Tasks = 797987 / 2500 = 319.19
# Calculated Tasks = 797987 / 1000 = 798
#SBATCH --array=0-79  # Create 80 tasks
##SBATCH --array=0-159  # Create 160 tasks
##SBATCH --array=0-319  # Create 320 tasks
##SBATCH --array=0-797
##SBATCH --mem=64G
##SBATCH --mem=128G
##SBATCH --partition=shared

source sbatch_env_setup.sh

# Create a Python script that handles the parallelized processing
python partition_dataset.py aleutian_islands --batch-size 10000 --batch-num $SLURM_ARRAY_TASK_ID
