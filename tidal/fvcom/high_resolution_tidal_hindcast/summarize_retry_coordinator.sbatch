#!/bin/bash
#SBATCH --account=hindcastra
#SBATCH --output=%x_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=andrew.simms@nrel.gov
#SBATCH --time=12:00:00
#SBATCH --partition=standard
#SBATCH --ntasks=1
#SBATCH --mem=8G

# Retry coordinator job
# Checks for missing batch files and submits retries to standard partition
# Uses --wait so this job blocks until all retries complete

# Get parameters from environment
LOCATION=${LOCATION:-cook_inlet}

# Source shared environment setup
source sbatch_env_setup.sh

# Run the coordinator script
python summarize_retry_coordinator.py $LOCATION

exit_code=$?

echo "Summarize retry completed with exit code $exit_code"

exit $exit_code
