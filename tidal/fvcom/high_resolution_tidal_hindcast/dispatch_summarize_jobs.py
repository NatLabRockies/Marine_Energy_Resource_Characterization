#!/usr/bin/env python3
"""
Master dispatch script: run_all_locations.py
Location configurations with temporal resolution mapping:
- 'hourly' temporal resolution -> batch size 10000
- 'half_hourly' temporal resolution -> batch size 5000
"""

import subprocess
import math

# Location configurations: location_name: {'faces': count, 'temporal_resolution': 'hourly'/'half_hourly'}
LOCATIONS = {
    # "aleutian_islands": {
    #     "faces": 797978,
    #     "temporal_resolution": "hourly",
    #     "process_runtime_hours": 2,
    # },
    # "cook_inlet": {
    #     "faces": 392002,
    #     "temporal_resolution": "hourly",
    #     "process_runtime_hours": 2,
    # },
    # "piscataqua_river": {
    #     "faces": 292927,
    #     "temporal_resolution": "half_hourly",
    #     "process_runtime_hours": 2,
    # },
    "puget_sound": {
        "faces": 1734765,
        "temporal_resolution": "half_hourly",
        # This is 73 half hourly files and is relatively slow
        "process_runtime_hours": 4,
    },
    # "western_passage": {
    #     "faces": 231208,
    #     "temporal_resolution": "half_hourly",
    #     "process_runtime_hours": 1,
    # },
}

# Mapping temporal resolution to batch size
BATCH_SIZE_MAP = {"hourly": 10000, "half_hourly": 5000}


def calculate_array_size(faces, batch_size):
    """Calculate array size needed (0-based indexing)"""
    return math.ceil(faces / batch_size) - 1


def submit_sbatch(args):
    """Submit sbatch job and return job ID"""
    result = subprocess.run(
        ["sbatch", "--parsable"] + args, capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


# Submit jobs for each location
for location, config in LOCATIONS.items():
    faces = config["faces"]
    temporal_resolution = config["temporal_resolution"]
    batch_size = BATCH_SIZE_MAP[temporal_resolution]
    array_size = calculate_array_size(faces, batch_size)
    process_runtime_hours = config["process_runtime_hours"]

    print(f"Submitting pipeline for {location}:")
    print(
        f"  faces={faces}, temporal_resolution={temporal_resolution}, batch_size={batch_size}, array=0-{array_size}"
    )

    # Submit the processing job array directly
    print(f"Submitting parallel processing jobs for {location}...")

    process_args = [
        f"--export=LOCATION={location},FACES={faces},BATCH_SIZE={batch_size}",
        f"--array=0-{array_size}",
        f"--output={location}_process_%A_%a.out",
        f"--job-name={location}_process",
        # Time in minutes
        f"--time={process_runtime_hours * 60}",
        "summarize_single_location_batch.sbatch",
    ]

    process_job_id = submit_sbatch(process_args)
    print(f"Process job array submitted with ID: {process_job_id}")

    # Submit the concatenation job that depends on the processing jobs
    print(f"Submitting concatenation job for {location}...")

    concat_args = [
        f"--dependency=afterok:{process_job_id}",
        f"--export=LOCATION={location}",
        f"--output={location}_concat_%j.out",
        f"--job-name={location}_concat",
        "summarize_location_concat.sbatch",
    ]

    concat_job_id = submit_sbatch(concat_args)
    print(f"Concatenation job submitted with ID: {concat_job_id}")
    print(f"All jobs submitted successfully for {location}!")
    # print(f"Process jobs: {process_job_id}")
    print(f"Concat job: {concat_job_id}")
    print("---")

print("All location pipelines submitted!")
