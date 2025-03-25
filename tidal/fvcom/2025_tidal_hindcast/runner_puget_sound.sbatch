#!/bin/bash
#SBATCH --account=hindcastra
#SBATCH --output=runner_puget_sound_%j.out
#SBATCH --job-name=runner_puget_sound
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=andrew.simms@nrel.gov
#SBATCH --time=120:00:00 # 5 days
#SBATCH --mem=230G # Max for non bigmem job
##SBATCH --partition=debug
##SBATCH --ntasks=12
##SBATCH --nodes=1
##SBATCH --time=59

# Load necessary modules (if any)
module load anaconda3/2023.07-2

conda init
conda activate tidal_fvcom

# Change to the directory containing the script
cd /home/asimms/marine_energy_resource_characterization/tidal/fvcom/2025_tidal_hindcast

# Force Python to flush print statements immediately rather than buffering them
# This ensures real-time output of print statements
export PYTHONUNBUFFERED=1

# Run the Python script with the desired arguments
python3 runner.py puget_sound
