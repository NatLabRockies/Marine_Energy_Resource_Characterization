#!/bin/bash
#SBATCH --account=hindcastra
#SBATCH --output=generate_manifest_%j.out
#SBATCH --job-name=generate_manifest
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=andrew.simms@nrel.gov
#SBATCH --time=8:00:00
#SBATCH --partition=shared
#SBATCH --mem=64G

# Prepare for python script execution
source sbatch_env_setup.sh

# Run the manifest generation script
python3 generate_parquet_partition_manifest_json.py
