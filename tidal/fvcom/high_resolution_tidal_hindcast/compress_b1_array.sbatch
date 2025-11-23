#!/bin/bash
#SBATCH --account=hindcastra
#SBATCH --output=%x_%A_%a.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=andrew.simms@nrel.gov
#SBATCH --time=24:00:00
#SBATCH --partition=standard

# SLURM array job for parallel b1_vap compression
# Environment variables required:
#   LOCATION: Location key (e.g., cook_inlet, aleutian_islands)
#   SKIP_EXISTING: 1 to skip existing files, 0 to reprocess (default: 0)

# Source shared environment setup
source sbatch_env_setup.sh

# Validate required environment variables
if [ -z "$LOCATION" ]; then
    echo "Error: LOCATION environment variable not set"
    exit 1
fi

# Set defaults
SKIP_EXISTING=${SKIP_EXISTING:-0}

# Print job information
echo "=========================================="
echo "B1 Compression Array Job"
echo "=========================================="
echo "Job ID:          $SLURM_JOB_ID"
echo "Array Task ID:   $SLURM_ARRAY_TASK_ID"
echo "Location:        $LOCATION"
echo "Skip existing:   $SKIP_EXISTING"
echo "=========================================="
echo ""

# Build command arguments
if [ "$SKIP_EXISTING" = "1" ]; then
    ARGS="$LOCATION $SLURM_ARRAY_TASK_ID --skip-existing"
else
    ARGS="$LOCATION $SLURM_ARRAY_TASK_ID"
fi

# Run the worker script
echo "Starting compression for file index $SLURM_ARRAY_TASK_ID..."
python compress_single_b1_file.py $ARGS

# Capture exit code
EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "Compression completed successfully"
else
    echo "Compression failed with exit code: $EXIT_CODE"
fi
echo "=========================================="

exit $EXIT_CODE
