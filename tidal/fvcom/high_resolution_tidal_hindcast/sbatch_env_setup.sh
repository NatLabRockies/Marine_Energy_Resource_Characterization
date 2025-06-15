#!/bin/bash
# Shared environment setup for Cook Inlet processing jobs

echo "Initializing tidal hindcast sbatch environment..."

# Load necessary modules
module load anaconda3/2023.07-2
conda init
conda activate tidal_fvcom

# Change to the directory containing the script
cd /home/asimms/marine_energy_resource_characterization/tidal/fvcom/high_resolution_tidal_hindcast

# Force Python to flush print statements immediately
export PYTHONUNBUFFERED=1

echo "Environment setup complete."
