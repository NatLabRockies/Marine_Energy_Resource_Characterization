#!/bin/bash
#SBATCH --account=hindcastra
#SBATCH --output=sum_cook_dispatch_%j.out
#SBATCH --job-name=sum_cook_dispatch
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=andrew.simms@nrel.gov
#SBATCH --time=01:00:00
#SBATCH --partition=standard

# Source shared environment setup
source sbatch_env_setup.sh

echo "Starting Cook Inlet processing pipeline..."

# Submit the parallel processing job array
echo "Submitting parallel processing jobs..."
PROCESS_JOB_ID=$(sbatch --parsable summarize_cook_inlet_process.sbatch)
echo "Process job array submitted with ID: $PROCESS_JOB_ID"

# Submit the concatenation job that depends on the processing jobs
echo "Submitting concatenation job..."
CONCAT_JOB_ID=$(sbatch --parsable --dependency=afterok:$PROCESS_JOB_ID summarize_cook_inlet_concat.sbatch)
echo "Concatenation job submitted with ID: $CONCAT_JOB_ID"

echo "All jobs submitted successfully!"
echo "Process jobs: $PROCESS_JOB_ID"
echo "Concat job: $CONCAT_JOB_ID"
echo "Pipeline complete - concatenation will run after all processing jobs finish."
