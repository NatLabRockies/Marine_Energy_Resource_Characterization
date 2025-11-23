#!/bin/bash
#SBATCH --account=hindcastra
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=andrew.simms@nrel.gov
#SBATCH --output=s3_upload_%A_%a.out
#SBATCH --job-name=s3_upload
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=shared
#SBATCH --time=12:00::00
#SBATCH --mem=32GB

# SLURM array job for S3 uploads
# Environment variables required:
#   MANIFEST_PATH: Path to SQLite manifest file
#   FILES_PER_JOB: Number of files to process per job
#   VERIFY_CHECKSUM: "true" or "false"

# Prepare for python script execution
source sbatch_env_setup.sh

# Validate required environment variables
if [ -z "$MANIFEST_PATH" ]; then
    echo "Error: MANIFEST_PATH environment variable not set"
    exit 1
fi

if [ -z "$FILES_PER_JOB" ]; then
    echo "Error: FILES_PER_JOB environment variable not set"
    exit 1
fi

# Set defaults
VERIFY_CHECKSUM=${VERIFY_CHECKSUM:-"true"}

# Print job information
echo "=========================================="
echo "S3 Upload Array Job Information"
echo "=========================================="
echo "Job ID:          $SLURM_JOB_ID"
echo "Array Task ID:   $SLURM_ARRAY_TASK_ID"
echo "Manifest:        $MANIFEST_PATH"
echo "Files per job:   $FILES_PER_JOB"
echo "Verify checksum: $VERIFY_CHECKSUM"
echo "Attempt:         $SLURM_RESTART_COUNT / 3"
echo "=========================================="
echo ""

# Build command arguments
ARGS="$MANIFEST_PATH $SLURM_ARRAY_TASK_ID $FILES_PER_JOB"

if [ "$VERIFY_CHECKSUM" = "false" ]; then
    ARGS="$ARGS --no-verify-checksum"
fi

# Run the upload script
python3 upload_from_manifest.py $ARGS

# Capture exit code
EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "Batch upload completed successfully"
else
    echo "Batch upload failed with exit code: $EXIT_CODE"
fi
echo "=========================================="

exit $EXIT_CODE
