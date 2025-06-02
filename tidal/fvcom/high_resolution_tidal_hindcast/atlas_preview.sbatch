#!/bin/bash
#SBATCH --account=hindcastra
#SBATCH --output=atlas_preview_%j.out
#SBATCH --job-name=atlas_preview
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=andrew.simms@nrel.gov
#SBATCH --time=59
#SBATCH --partition=debug


# Load necessary modules (if any)
module load anaconda3/2023.07-2

conda init
conda activate tidal_fvcom

# Change to the directory containing the script
cd /home/asimms/marine_energy_resource_characterization/tidal/fvcom/high_resolution_tidal_hindcast

# Force Python to flush print statements immediately rather than buffering them
# This ensures real-time output of print statements
export PYTHONUNBUFFERED=1

# Run the Python script with the desired arguments
python3 atlas_preview.py
