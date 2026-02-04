#!/bin/bash
#SBATCH --account=hindcastra
#SBATCH --output=cook_hsds_%j.out
#SBATCH --job-name=cook_hsds
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=andrew.simms@nrel.gov
#SBATCH --time=120:00:00
#SBATCH --partition=long

# Source shared environment setup
source sbatch_env_setup.sh

python convert_tidal_data_for_hsds.py
