#!/bin/bash
#SBATCH --account=hindcastra
#SBATCH --output=extract_point_%j.out
#SBATCH --job-name=extract_point
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=andrew.simms@nrel.gov
#SBATCH --time=2870 # 2 days, 23 hours, 50 minutes
#SBATCH --partition=standard

# Source shared environment setup
source sbatch_env_setup.sh

# Run the Python script with the desired arguments
python3 extract_point.py
