#!/bin/bash
#SBATCH --account=hindcastra
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=andrew.simms@nrel.gov
#SBATCH --output=s3_upload_%j.out
#SBATCH --job-name=s3_upload
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=shared
#SBATCH --mem=32GB

# SLURM batch script for uploading single files to S3
# This script is called by dispatch_s3_upload.py with environment variables:
#   LOCAL_FILE: Full path to local file
#   S3_DESTINATION: S3 destination path (e.g., AK_cook_inlet/0.4.0/b1_vap/2005/file.nc)
#   SKIP_IF_EXISTS: "true" or "false"
#   VERIFY_CHECKSUM: "true" or "false"

# Prepare for python script execution
source sbatch_env_setup.sh

# Validate required environment variables
if [ -z "$LOCAL_FILE" ]; then
    echo "Error: LOCAL_FILE environment variable not set"
    exit 1
fi

if [ -z "$S3_DESTINATION" ]; then
    echo "Error: S3_DESTINATION environment variable not set"
    exit 1
fi

# Set defaults if not provided
SKIP_IF_EXISTS=${SKIP_IF_EXISTS:-"true"}
VERIFY_CHECKSUM=${VERIFY_CHECKSUM:-"true"}

# Build command line arguments
ARGS="$LOCAL_FILE $S3_DESTINATION"

if [ "$SKIP_IF_EXISTS" = "false" ]; then
    ARGS="$ARGS --no-skip-if-exists"
fi

if [ "$VERIFY_CHECKSUM" = "false" ]; then
    ARGS="$ARGS --no-verify-checksum"
fi

# Print job information
echo "=========================================="
echo "S3 Upload Job Information"
echo "=========================================="
echo "Job ID:          $SLURM_JOB_ID"
echo "Local file:      $LOCAL_FILE"
echo "S3 destination:  $S3_DESTINATION"
echo "Skip if exists:  $SKIP_IF_EXISTS"
echo "Verify checksum: $VERIFY_CHECKSUM"
echo "=========================================="
echo ""

# Run the upload script
python3 upload_to_s3.py $ARGS

# Capture exit code
EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "Upload completed successfully"
else
    echo "Upload failed with exit code: $EXIT_CODE"
fi
echo "=========================================="

exit $EXIT_CODE
