#!/bin/bash
#SBATCH --account=hindcastra
#SBATCH --output=runner_puget_sound_%j.out
#SBATCH --job-name=runner_puget_sound
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=andrew.simms@nrel.gov
##SBATCH --time=120:00:00 # 5 days
#SBATCH --time=47:00:00 # Max for standard job
#SBATCH --partition=standard
##SBATCH --mem=230G # Max for non bigmem job
##SBATCH --time=59:00
##SBATCH --partition=debug

# Prepare for python script execution
source sbatch_env_setup.sh

# Run the Python script with the desired arguments
python3 runner.py puget_sound
