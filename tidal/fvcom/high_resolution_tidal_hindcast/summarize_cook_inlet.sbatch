#!/bin/bash
#SBATCH --account=hindcastra
#SBATCH --output=sum_cook_%A_%a.out
##SBATCH --output=sum_cook_%j.out
#SBATCH --job-name=sum_cook
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=andrew.simms@nrel.gov
##SBATCH --time=24:00:00
#SBATCH --time=59:00
##SBATCH --partition=standard
#SBATCH --partition=debug
# Faces 392002
# Time Interval: Hourly
# Recommended Batch Size: 5000
# Calculated Tasks = 392002 / 5000 = 78.4004
##SBATCH --array=0-78  # Create 79 tasks
# SBATCH --array=0-2  # For testing, use 2 tasks

# Load necessary modules
module load anaconda3/2023.07-2
conda init
conda activate tidal_fvcom

# Change to the directory containing the script
cd /home/asimms/marine_energy_resource_characterization/tidal/fvcom/high_resolution_tidal_hindcast

# Force Python to flush print statements immediately
export PYTHONUNBUFFERED=1

# Create a Python script that handles the parallelized processing
python summarize_dataset.py cook_inlet --batch-size 5000 --batch-num $SLURM_ARRAY_TASK_ID 
# python summarize_dataset.py cook_inlet --batch-size 5000 --batch-num $SLURM_ARRAY_TASK_ID

# Create a completion marker for this task
mkdir -p /tmp/cook_completion_markers
touch /tmp/cook_completion_markers/task_${SLURM_ARRAY_TASK_ID}_done

# Only the last task handles concatenation
if [ $SLURM_ARRAY_TASK_ID -eq 78 ]; then
    echo "Waiting for all tasks to complete..."
    
    # Wait until all 79 completion markers exist
    while [ $(ls /tmp/cook_completion_markers/task_*_done 2>/dev/null | wc -l) -lt 79 ]; do
        sleep 10  # check every 10 seconds
    done
    
    echo "All summarize jobs complete. Starting concatenation..."
    
    python concat_summarized_datasets.py cook_inlet --batch-num 0  # 0 does monthly files
    # Run both concat jobs in parallel
    # python concat_summarized_datasets.py cook_inlet --batch-num 0 &  # 0 does monthly files
    # python concat_summarized_datasets.py cook_inlet --batch-num 1 &  # 1 does yearly files
    # wait
    
    # Cleanup
    rm -rf /tmp/cook_completion_markers
    echo "All jobs completed!"
fi
