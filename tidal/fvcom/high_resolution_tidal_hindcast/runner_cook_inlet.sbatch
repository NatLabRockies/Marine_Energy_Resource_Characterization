#!/bin/bash
#SBATCH --account=hindcastra
#SBATCH --output=runner_cook_inlet_%j.out
#SBATCH --job-name=runner_cook_inlet
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=andrew.simms@nrel.gov
##SBATCH --time=12:00:00
##SBATCH --partition=standard
#SBATCH --time=59
#SBATCH --partition=debug

# Prepare for python script execution
source sbatch_env_setup.sh

# Run the Python script with the desired arguments
python3 runner.py cook_inlet
