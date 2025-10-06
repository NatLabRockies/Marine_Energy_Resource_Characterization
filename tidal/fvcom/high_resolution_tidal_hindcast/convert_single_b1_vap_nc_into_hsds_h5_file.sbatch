#!/bin/bash
#SBATCH --account=hindcastra
#SBATCH --output=convert_b1_hsds_%A_%a.out
#SBATCH --job-name=convert_b1_hsds
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=andrew.simms@nrel.gov
#SBATCH --partition=shared
#SBATCH --mem=128GB
#SBATCH --time=4:00:00

# Prepare for python script execution
source sbatch_env_setup.sh

# Run the conversion script with the chunk index from SLURM_ARRAY_TASK_ID
python3 convert_b1_vap_tidal_data_for_hsds.py --chunk ${SLURM_ARRAY_TASK_ID} --location ${LOCATION}