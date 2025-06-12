#!/bin/bash
#SBATCH --account=hindcastra
#SBATCH --output=sum_cook%A_%a.out
#SBATCH --job-name=sum_cook
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=andrew.simms@nrel.gov
##SBATCH --time=24:00:00
##SBATCH --time=59:00
##SBATCH --partition=standard
#SBATCH --partition=debug
# Faces 392002
# Time Interval: Hourly
# Recommended Batch Size: 5000
# Calculated Tasks = 392002 / 5000 = 78.4004
##SBATCH --array=0-78  # Create 79 tasks

# Load necessary modules
module load anaconda3/2023.07-2
conda init
conda activate tidal_fvcom

# Change to the directory containing the script
cd /home/asimms/marine_energy_resource_characterization/tidal/fvcom/high_resolution_tidal_hindcast

# Force Python to flush print statements immediately
export PYTHONUNBUFFERED=1

# Create a Python script that handles the parallelized processing
# python summarize_dataset.py cook_inlet --batch-size 5000 --batch-num $SLURM_ARRAY_TASK_ID 
python summarize_dataset.py cook_inlet --batch-size 5 --batch-num 0
