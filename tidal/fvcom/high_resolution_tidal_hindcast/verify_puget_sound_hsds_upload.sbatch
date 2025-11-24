#!/bin/bash
#SBATCH --account=hindcastra
#SBATCH --output=verify_s3_puget_sound_%j.out
#SBATCH --job-name=verify_s3_puget
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=andrew.simms@nrel.gov
#SBATCH --time=24:00:00
#SBATCH --partition=shared
#SBATCH --mem=64G

source sbatch_env_setup.sh

python verify_s3_upload.py --bucket oedi-data-drop --key us-tidal/WA_puget_sound/v1.0.0/hsds/WA_puget_sound.wpto_high_res_tidal.hsds.v1.0.0.h5 --local-path /projects/hindcastra/Tidal/datasets/high_resolution_tidal_hindcast/WA_puget_sound/v1.0.0/hsds/WA_puget_sound.wpto_high_res_tidal.hsds.v1.0.0.h5
