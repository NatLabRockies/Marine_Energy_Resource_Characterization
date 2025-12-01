#!/bin/bash
#SBATCH --account=hindcastra
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=andrew.simms@nrel.gov
#SBATCH --output=cache/s3_delete/logs/s3_delete_%A_%a.out
#SBATCH --job-name=s3_delete
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=shared
#SBATCH --time=1:00:00
#SBATCH --mem=4GB

# SLURM array job for S3 deletions
# Environment variables required:
#   DELETE_MANIFEST: Path to JSON manifest file
#   FILES_PER_JOB: Number of files to delete per job
#   S3_BUCKET: S3 bucket name
#   S3_PROFILE: AWS profile name

# Prepare for python script execution
source sbatch_env_setup.sh

# Create log directory if needed
mkdir -p cache/s3_delete/logs

# Validate required environment variables
if [ -z "$DELETE_MANIFEST" ]; then
    echo "Error: DELETE_MANIFEST environment variable not set"
    exit 1
fi

if [ -z "$FILES_PER_JOB" ]; then
    echo "Error: FILES_PER_JOB environment variable not set"
    exit 1
fi

if [ -z "$S3_BUCKET" ]; then
    echo "Error: S3_BUCKET environment variable not set"
    exit 1
fi

if [ -z "$S3_PROFILE" ]; then
    echo "Error: S3_PROFILE environment variable not set"
    exit 1
fi

# Print job information
echo "=========================================="
echo "S3 Delete Array Job Information"
echo "=========================================="
echo "Job ID:          $SLURM_JOB_ID"
echo "Array Task ID:   $SLURM_ARRAY_TASK_ID"
echo "Manifest:        $DELETE_MANIFEST"
echo "Files per job:   $FILES_PER_JOB"
echo "S3 Bucket:       $S3_BUCKET"
echo "S3 Profile:      $S3_PROFILE"
echo "=========================================="
echo ""

# Run the delete worker script
python3 s3_delete_worker.py \
    "$DELETE_MANIFEST" \
    "$SLURM_ARRAY_TASK_ID" \
    "$FILES_PER_JOB" \
    --bucket "$S3_BUCKET" \
    --profile "$S3_PROFILE"

# Capture exit code
EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "Batch delete completed successfully"
else
    echo "Batch delete failed with exit code: $EXIT_CODE"
fi
echo "=========================================="

exit $EXIT_CODE
