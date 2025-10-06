#!/usr/bin/env python3
"""
Dispatch script for converting b1_vap data to HSDS format.

This script:
1. Counts the number of b1_vap temporal chunk files for the specified location
2. Submits a SLURM array job to convert each chunk to HSDS format in parallel
3. Submits a dependent job to stitch the individual HSDS files into a single yearly file

Usage:
    python dispatch_b1_to_hsds_jobs.py --location <location_name>

Example:
    python dispatch_b1_to_hsds_jobs.py --location cook_inlet
"""

import argparse
import subprocess
from pathlib import Path

from config import config
from src.file_manager import get_vap_output_dir

# Location-specific resource configurations
# Each location has different resource requirements based on:
# - Number of temporal chunks (hourly vs half-hourly resolution)
# - File sizes
# - Processing complexity
LOCATION_RESOURCES = {
    "aleutian_islands": {
        "convert": {
            "partition": "shared",
            "mem": "128GB",
            "time": "4:00:00",
        },
        "stitch": {
            "partition": "standard",
            "mem": "256GB",
            "time": "6:00:00",
        },
    },
    "cook_inlet": {
        "convert": {
            "partition": "shared",
            "mem": "128GB",
            "time": "4:00:00",
        },
        "stitch": {
            "partition": "standard",
            "mem": "128GB",
            "time": "4:00:00",
        },
    },
    "piscataqua_river": {
        "convert": {
            "partition": "shared",
            "mem": "128GB",
            "time": "4:00:00",
        },
        "stitch": {
            "partition": "standard",
            "mem": "128GB",
            "time": "4:00:00",
        },
    },
    "puget_sound": {
        "convert": {
            "partition": "shared",
            "mem": "128GB",
            "time": "6:00:00",
        },
        "stitch": {
            "partition": "standard",
            "mem": "256GB",
            "time": "8:00:00",
        },
    },
    "western_passage": {
        "convert": {
            "partition": "shared",
            "mem": "128GB",
            "time": "4:00:00",
        },
        "stitch": {
            "partition": "standard",
            "mem": "128GB",
            "time": "4:00:00",
        },
    },
}


def submit_sbatch(args):
    """Submit sbatch job and return job ID"""
    result = subprocess.run(
        ["sbatch", "--parsable"] + args, capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def count_vap_files(location_name):
    """Count the number of b1_vap temporal chunk files for the location"""
    location_config = config["location_specification"][location_name]
    vap_dir = get_vap_output_dir(config, location_config)

    # Find all .nc files in the vap directory
    nc_files = sorted(list(vap_dir.rglob("*.nc")))

    if not nc_files:
        raise ValueError(f"No .nc files found in {vap_dir}")

    return len(nc_files)


def main():
    parser = argparse.ArgumentParser(
        description="Dispatch HSDS conversion jobs for a location"
    )
    parser.add_argument(
        "--location",
        type=str,
        required=True,
        choices=list(config["location_specification"].keys()),
        help="Location name from config.py",
    )
    args = parser.parse_args()

    location = args.location

    # Validate location has resource configuration
    if location not in LOCATION_RESOURCES:
        raise ValueError(
            f"Location {location} not found in LOCATION_RESOURCES. "
            f"Available: {list(LOCATION_RESOURCES.keys())}"
        )

    # Count the number of temporal chunk files
    num_chunks = count_vap_files(location)
    print(f"\nSubmitting HSDS conversion pipeline for {location}:")
    print(f"  Found {num_chunks} b1_vap temporal chunk files")
    print(f"  Array job will process chunks 0-{num_chunks - 1}")

    # Get resource configurations for this location
    convert_resources = LOCATION_RESOURCES[location]["convert"]
    stitch_resources = LOCATION_RESOURCES[location]["stitch"]

    # Submit the convert array job
    print(f"\nSubmitting parallel conversion jobs for {location}...")
    convert_args = [
        f"--export=LOCATION={location}",
        f"--array=0-{num_chunks - 1}",
        f"--partition={convert_resources['partition']}",
        f"--mem={convert_resources['mem']}",
        f"--time={convert_resources['time']}",
        f"--output=convert_b1_hsds_{location}_%A_%a.out",
        f"--job-name=convert_hsds_{location}",
        "convert_single_b1_vap_nc_into_hsds_h5_file.sbatch",
    ]

    convert_job_id = submit_sbatch(convert_args)
    print(f"Convert array job submitted with ID: {convert_job_id}")

    # Submit the stitch job that depends on the convert array job
    print(f"\nSubmitting stitching job for {location}...")
    stitch_args = [
        f"--dependency=afterok:{convert_job_id}",
        f"--export=LOCATION={location}",
        f"--partition={stitch_resources['partition']}",
        f"--mem={stitch_resources['mem']}",
        f"--time={stitch_resources['time']}",
        f"--output=stitch_hsds_{location}_%j.out",
        f"--job-name=stitch_hsds_{location}",
        "stitch_prepared_b1_files_into_singular_hsds_h5_file.sbatch",
    ]

    stitch_job_id = submit_sbatch(stitch_args)
    print(f"Stitch job submitted with ID: {stitch_job_id}")

    # Print monitoring information
    print(f"\n{'=' * 60}")
    print(f"All jobs submitted successfully for {location}!")
    print(f"{'=' * 60}")
    print(f"\nConvert array job: {convert_job_id}")
    print(f"  - Processing {num_chunks} chunks in parallel")
    print(f"  - Partition: {convert_resources['partition']}")
    print(f"  - Memory: {convert_resources['mem']}")
    print(f"  - Time limit: {convert_resources['time']}")
    print(f"\nStitch job: {stitch_job_id}")
    print("  - Will run after convert job completes")
    print(f"  - Partition: {stitch_resources['partition']}")
    print(f"  - Memory: {stitch_resources['mem']}")
    print(f"  - Time limit: {stitch_resources['time']}")
    print("\nMonitoring commands:")
    print("  squeue -u $USER")
    print(f"  squeue -j {convert_job_id}")
    print(f"  squeue -j {stitch_job_id}")
    print(f"  tail -f convert_b1_hsds_{location}_{convert_job_id}_*.out")
    print(f"  tail -f stitch_hsds_{location}_{stitch_job_id}.out")
    print()


if __name__ == "__main__":
    main()
