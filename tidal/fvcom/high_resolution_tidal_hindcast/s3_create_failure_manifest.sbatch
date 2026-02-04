#!/bin/bash
#SBATCH --account=hindcastra
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=andrew.simms@nrel.gov
#SBATCH --output=s3_failure_manifest_%j.out
#SBATCH --job-name=s3_failure_manifest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=short
## SBATCH --mem=8GB
#SBATCH --time=10

# Create failure manifest after all upload jobs complete
# Environment variables required:
#   MANIFEST_PATH: Path to SQLite manifest file

# Prepare for python script execution
source sbatch_env_setup.sh

# Validate required environment variables
if [ -z "$MANIFEST_PATH" ]; then
    echo "Error: MANIFEST_PATH environment variable not set"
    exit 1
fi

# Print job information
echo "=========================================="
echo "Failure Manifest Creation"
echo "=========================================="
echo "Job ID:          $SLURM_JOB_ID"
echo "Manifest:        $MANIFEST_PATH"
echo "=========================================="
echo ""

# Run the failure manifest script
python3 s3_create_failure_manifest.py "$MANIFEST_PATH"

# Capture exit code
EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "No failures - all uploads successful!"
elif [ $EXIT_CODE -eq 1 ]; then
    echo "Failure manifest created - check output"
else
    echo "Error creating failure manifest"
fi
echo "=========================================="

exit $EXIT_CODE
