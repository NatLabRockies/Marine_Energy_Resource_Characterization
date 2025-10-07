#!/bin/bash
#SBATCH --account=hindcastra
#SBATCH --output=stitch_hsds_%j.out
#SBATCH --job-name=stitch_hsds
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=andrew.simms@nrel.gov
##SBATCH --partition=standard
##SBATCH --time=47:00:00
#SBATCH --partition=debug
#SBATCH --time=59

# Prepare for python script execution
source sbatch_env_setup.sh

# Run the stitching script
python3 stitch_individual_hsds_tidal_data_into_single_file.py --location ${LOCATION}
