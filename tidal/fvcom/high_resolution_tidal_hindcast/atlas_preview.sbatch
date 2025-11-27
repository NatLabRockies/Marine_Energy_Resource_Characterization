#!/bin/bash
#SBATCH --account=hindcastra
#SBATCH --output=atlas_preview_%j.out
#SBATCH --job-name=atlas_preview
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=andrew.simms@nrel.gov
##SBATCH --time=59
##SBATCH --partition=debug
#SBATCH --time=3:59:59
#SBATCH --partition=short

source sbatch_env_setup.sh

# Run the Python script with the desired arguments
python3 atlas_preview.py
