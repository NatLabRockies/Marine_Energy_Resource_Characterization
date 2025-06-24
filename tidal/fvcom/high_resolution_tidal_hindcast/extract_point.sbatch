#!/bin/bash
#SBATCH --account=hindcastra
#SBATCH --output=runner_cook_inlet_%j.out
#SBATCH --job-name=runner_cook_inlet
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=andrew.simms@nrel.gov
#SBATCH --time=59
#SBATCH --partition=debug

# Source shared environment setup
source sbatch_env_setup.sh

# Run the Python script with the desired arguments
python3 extract_point.py
