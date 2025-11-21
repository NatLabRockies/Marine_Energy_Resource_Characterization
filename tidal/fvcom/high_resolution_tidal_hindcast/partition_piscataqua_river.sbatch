#!/bin/bash
#SBATCH --account=hindcastra
#SBATCH --output=part_pisc_%A_%a.out
#SBATCH --job-name=part_pisc
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=andrew.simms@nrel.gov
##SBATCH --time=8:00:00
#SBATCH --time=3:59:00
#SBATCH --partition=short
##SBATCH --mem=128G
# Faces 292927
# Time Interval: Half Hourly
# Recommended Batch Size: 2500
# Recommended Batch Size: 1000
# Calculated Tasks = 292927 / 2500 = 117.17
# Calculated Tasks = 292927 / 1000 = 293
#SBATCH --array=0-117  # Create 118 tasks

source sbatch_env_setup.sh

# Change to the directory containing the script
cd /home/asimms/marine_energy_resource_characterization/tidal/fvcom/high_resolution_tidal_hindcast

# Force Python to flush print statements immediately
export PYTHONUNBUFFERED=1

# Create a Python script that handles the parallelized processing
python partition_dataset.py piscataqua_river --batch-size 2500 --batch-num $SLURM_ARRAY_TASK_ID 
