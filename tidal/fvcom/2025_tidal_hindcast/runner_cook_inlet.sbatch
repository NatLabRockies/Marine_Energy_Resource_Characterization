#!/bin/bash
#SBATCH --account=hindcastra
#SBATCH --output=runner_cook_inlet_%j.out
#SBATCH --error=runner_cook_inlet_%j.err
#SBATCH --job-name=runner_cook_inlet
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=andrew.simms@nrel.gov
#SBATCH --time=06:00:00

# Load necessary modules (if any)
module load anaconda3/2023.07-2

conda activate tidal_fvcom

# Change to the directory containing the script
cd /home/asimms/marine_energy_resource_characterization/tidal/fvcom/2025_tidal_hindcast

# Run the Python script with the desired arguments
python3 runner.py cook_inlet
