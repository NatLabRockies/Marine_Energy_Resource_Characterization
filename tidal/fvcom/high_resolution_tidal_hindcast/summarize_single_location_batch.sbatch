#!/bin/bash
#SBATCH --account=hindcastra
#SBATCH --output=%x_%A_%a.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=andrew.simms@nrel.gov
#SBATCH --time=12:00:00
#SBATCH --partition=standard
##SBATCH --ntasks=1
##SBATCH --mem=128

# Get parameters from environment
LOCATION=${LOCATION:-cook_inlet}
BATCH_SIZE=${BATCH_SIZE:-10000}

# Source shared environment setup
source sbatch_env_setup.sh

echo "Starting processing task $SLURM_ARRAY_TASK_ID for $LOCATION..."
echo "Location: $LOCATION, Batch size: $BATCH_SIZE, Task ID: $SLURM_ARRAY_TASK_ID"

# Run the Python script with location-specific parameters
python summarize_dataset.py $LOCATION --batch-size $BATCH_SIZE --batch-num $SLURM_ARRAY_TASK_ID

echo "Task $SLURM_ARRAY_TASK_ID for $LOCATION completed successfully."
