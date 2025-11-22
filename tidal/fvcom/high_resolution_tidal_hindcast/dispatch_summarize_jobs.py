#!/usr/bin/env python3
"""
Master dispatch script for submitting summarization jobs.
Location configurations with temporal resolution mapping:
- 'hourly' temporal resolution -> batch size 10000
- 'half_hourly' temporal resolution -> batch size 5000

Usage:
    python dispatch_summarize_jobs.py <location>

Example:
    python dispatch_summarize_jobs.py cook_inlet
    python dispatch_summarize_jobs.py puget_sound
"""

import subprocess
import math
import argparse
import sys

# Location configurations: location_name: {'faces': count, 'temporal_resolution': 'hourly'/'half_hourly'}
LOCATIONS = {
    "aleutian_islands": {
        "faces": 797978,
        "temporal_resolution": "hourly",
        "process_runtime_hours": 18,
        "retry_runtime_hours": 24,
    },
    "cook_inlet": {
        "faces": 392002,
        "temporal_resolution": "hourly",
        "process_runtime_hours": 4,
        "retry_runtime_hours": 6,
    },
    "piscataqua_river": {
        "faces": 292927,
        "temporal_resolution": "half_hourly",
        "process_runtime_hours": 4,
        "retry_runtime_hours": 6,
    },
    "puget_sound": {
        "faces": 1734765,
        "temporal_resolution": "half_hourly",
        # This is 73 half hourly files and is relatively slow
        # Short partition
        "process_runtime_hours": 18,
        "retry_runtime_hours": 24,
    },
    "western_passage": {
        "faces": 231208,
        "temporal_resolution": "half_hourly",
        "process_runtime_hours": 4,
        "retry_runtime_hours": 6,
    },
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


def validate_location(location):
    """Validate location argument."""
    if location not in LOCATIONS:
        valid_locations = list(LOCATIONS.keys())
        raise argparse.ArgumentTypeError(
            f"Invalid location: {location}. Must be one of: {', '.join(valid_locations)}"
        )
    return location


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Submit SLURM jobs for summarizing FVCOM data at a specific location."
    )

    parser.add_argument(
        "location",
        type=validate_location,
        help="Location to process (e.g., aleutian_islands, cook_inlet, piscataqua_river, puget_sound, western_passage)",
    )

    parser.add_argument(
        "--retry-indices",
        type=str,
        default=None,
        help="Comma-separated list of array indices to retry (e.g., '5,10,15' or '5-10,15,20-25'). Skips concat job.",
    )

    parser.add_argument(
        "--skip-concat",
        action="store_true",
        help="Skip the concatenation job (useful when retrying specific batches)",
    )

    return parser.parse_args()


def submit_location_jobs(location):
    """Submit processing, retry coordinator, and concatenation jobs for a specific location."""
    config = LOCATIONS[location]
    faces = config["faces"]
    temporal_resolution = config["temporal_resolution"]
    batch_size = BATCH_SIZE_MAP[temporal_resolution]
    array_size = calculate_array_size(faces, batch_size)
    process_runtime_hours = config["process_runtime_hours"]

    print(f"Submitting pipeline for {location}:")
    print(
        f"  faces={faces}, temporal_resolution={temporal_resolution}, batch_size={batch_size}, array=0-{array_size}"
    )

    # Stage 1: Submit the processing job array to shared partition
    print(
        f"Stage 1: Submitting parallel processing jobs for {location} (shared partition)..."
    )

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

    # Stage 2: Submit retry coordinator (runs after all shared jobs, regardless of success/fail)
    print(f"Stage 2: Submitting retry coordinator for {location}...")

    coordinator_args = [
        f"--dependency=afterany:{process_job_id}",
        f"--export=LOCATION={location}",
        f"--output={location}_coordinator_%j.out",
        f"--job-name={location}_coordinator",
        "summarize_retry_coordinator.sbatch",
    ]

    coordinator_job_id = submit_sbatch(coordinator_args)
    print(f"Coordinator job submitted with ID: {coordinator_job_id}")

    # Stage 3: Submit concatenation job (runs only if coordinator succeeds)
    print(f"Stage 3: Submitting concatenation job for {location}...")

    concat_args = [
        f"--dependency=afterok:{coordinator_job_id}",
        f"--export=LOCATION={location}",
        f"--output={location}_concat_%j.out",
        f"--job-name={location}_concat",
        "summarize_location_concat.sbatch",
    ]

    concat_job_id = submit_sbatch(concat_args)
    print(f"Concatenation job submitted with ID: {concat_job_id}")
    print(f"All jobs submitted successfully for {location}!")
    print(f"  Process jobs: {process_job_id}")
    print(f"  Coordinator: {coordinator_job_id}")
    print(f"  Concat: {concat_job_id}")
    print("---")


if __name__ == "__main__":
    args = parse_args()
    submit_location_jobs(args.location)
    print(f"Pipeline submitted for {args.location}!")
