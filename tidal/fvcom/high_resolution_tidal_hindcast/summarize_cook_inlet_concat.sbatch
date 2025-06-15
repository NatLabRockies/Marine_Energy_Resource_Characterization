#!/bin/bash
#SBATCH --account=hindcastra
#SBATCH --output=sum_cook_concat_%j.out
#SBATCH --job-name=sum_cook_concat
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=andrew.simms@nrel.gov
#SBATCH --time=02:00:00
#SBATCH --partition=standard

# Source shared environment setup
source sbatch_env_setup.sh

echo "Starting concatenation job..."
echo "All processing jobs have completed successfully (enforced by SLURM dependency)."

python concat_summarized_datasets.py cook_inlet --batch-num 0  # 0 does yearly files

# Optional: Run both concat jobs in parallel if needed
# python concat_summarized_datasets.py cook_inlet --batch-num 0 &  # 0 does yearly files
# python concat_summarized_datasets.py cook_inlet --batch-num 1 &  # 1 does monthly files
# wait

echo "All Cook Inlet processing jobs completed successfully!"
