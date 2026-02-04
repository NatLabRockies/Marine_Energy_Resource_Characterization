#!/bin/bash
#SBATCH --account=hindcastra
#SBATCH --output=sum_cook_test_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=andrew.simms@nrel.gov
#SBATCH --time=59
#SBATCH --partition=debug

# Source shared environment setup
source sbatch_env_setup.sh

# Run the Python script with location-specific parameters
python summarize_dataset.py cook_inlet --batch-size 10000 --batch-num 0
