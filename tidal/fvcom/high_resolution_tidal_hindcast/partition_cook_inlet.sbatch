#!/bin/bash
#SBATCH --account=hindcastra
#SBATCH --output=parquet_conversion_%A_%a.out
#SBATCH --job-name=parquet_convert
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=andrew.simms@nrel.gov
#SBATCH --time=24:00:00
#SBATCH --partition=standard
# There are ~400000 faces and batch size is 5000
#SBATCH --array=0-79  # Create 80 tasks (modify based on your total faces)

# Load necessary modules
module load anaconda3/2023.07-2
conda init
conda activate tidal_fvcom

# Change to the directory containing the script
cd /home/asimms/marine_energy_resource_characterization/tidal/fvcom/high_resolution_tidal_hindcast

# Force Python to flush print statements immediately
export PYTHONUNBUFFERED=1

# Create a Python script that handles the parallelized processing
python partition_dataset.py cook_inlet --batch-size 5000 --batch-number $SLURM_ARRAY_TASK_ID 
