#!/bin/bash
#SBATCH --account=hindcastra
#SBATCH --output=part_pisc_%A_%a.out
#SBATCH --job-name=part_pisc
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=andrew.simms@nrel.gov
#SBATCH --time=6:00:00
#SBATCH --partition=standard
##SBATCH --mem=128G
# Faces 292927
# Time Interval: Half Hourly
# Recommended Batch Size: 2500
# Recommended Batch Size: 1000
# Calculated Tasks = 292927 / 5000 = 58.59 -> ceil = 59 tasks (0-58)
# Calculated Tasks = 292927 / 2500 = 117.17 -> ceil = 118 tasks (0-117)
# Calculated Tasks = 292927 / 1000 = 293 -> ceil = 293 tasks (0-292)
##SBATCH --array=0-117  # Create 118 tasks
#SBATCH --array=0-58  # Create 59 tasks (faces 0-292926)

source sbatch_env_setup.sh

# Create a Python script that handles the parallelized processing
python partition_dataset.py piscataqua_river --batch-size 5000 --batch-num $SLURM_ARRAY_TASK_ID 
