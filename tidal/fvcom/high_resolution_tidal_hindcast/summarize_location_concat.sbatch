#!/bin/bash
#SBATCH --account=hindcastra
#SBATCH --output=%x_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=andrew.simms@nrel.gov
#SBATCH --time=24:00:00
#SBATCH --partition=standard

# Get parameters from environment
LOCATION=${LOCATION:-cook_inlet}

# Source shared environment setup
source sbatch_env_setup.sh

echo "Starting concatenation for $LOCATION..."

# Run your concatenation script here
python concat_summarized_datasets.py $LOCATION --batch-num 0
# python concatenate_results.py $LOCATION --batch-num 1

echo "Concatenation for $LOCATION completed successfully."
