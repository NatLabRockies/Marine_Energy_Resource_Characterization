#!/bin/bash
# Shared environment setup for tidal hindcast processing jobs

# Exit on error and propagate failures in pipelines
set -e
set -o pipefail

echo "Initializing tidal hindcast sbatch environment..."

# Load necessary modules (don't pin version - use system default)
module load conda

# Initialize conda for this shell (required for non-interactive SLURM jobs)
eval "$(conda shell.bash hook)"

# Activate the target environment
conda activate tidal_fvcom

# Change to the directory containing the script
cd /home/asimms/marine_energy_resource_characterization/tidal/fvcom/high_resolution_tidal_hindcast

# Force Python to flush print statements immediately
export PYTHONUNBUFFERED=1

echo "Environment setup complete."
