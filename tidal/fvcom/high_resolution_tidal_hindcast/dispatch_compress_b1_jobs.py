"""
Dispatch script for parallelized b1_vap compression jobs.

This script finds all b1_vap files for a location and submits SLURM array jobs
to process each file in parallel.

Usage:
    python dispatch_compress_b1_jobs.py <location> [--skip-existing]

Example:
    python dispatch_compress_b1_jobs.py cook_inlet
    python dispatch_compress_b1_jobs.py aleutian_islands --skip-existing
"""

import argparse
import subprocess
import sys
from pathlib import Path

from config import config
from src import file_manager


def validate_location(location):
    """Validate location argument."""
    if location not in config["location_specification"]:
        valid_locations = list(config["location_specification"].keys())
        raise argparse.ArgumentTypeError(
            f"Invalid location: {location}. Must be one of: {', '.join(valid_locations)}"
        )
    return location


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Submit SLURM array jobs for compressing b1_vap files in parallel."
    )

    parser.add_argument(
        "location",
        type=validate_location,
        help="Location to process (e.g., aleutian_islands, cook_inlet, piscataqua_river, puget_sound, western_passage)",
    )

    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip processing if output files already exist (default: reprocess all)",
    )

    return parser.parse_args()


def get_b1_files(location_key):
    """
    Get list of b1_vap files for location.

    Args:
        location_key: Location key from config

    Returns:
        Sorted list of b1_vap file paths
    """
    location = config["location_specification"][location_key]
    vap_dir = file_manager.get_vap_output_dir(config, location)

    # Find all b1_vap NetCDF files
    b1_files = sorted(vap_dir.glob("*.nc"))

    return b1_files


def submit_sbatch(args):
    """Submit sbatch job and return job ID."""
    result = subprocess.run(
        ["sbatch", "--parsable"] + args, capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def dispatch_jobs(location_key, skip_existing=False):
    """
    Dispatch parallel compression jobs for location.

    Args:
        location_key: Location key from config
        skip_existing: If True, skip files with existing outputs
    """
    location = config["location_specification"][location_key]

    print(f"Dispatching compression jobs for {location['label']}...")

    # Get b1_vap files
    b1_files = get_b1_files(location_key)

    if not b1_files:
        print(f"No b1_vap files found for {location_key}")
        print(
            f"Expected directory: {file_manager.get_vap_output_dir(config, location)}"
        )
        sys.exit(1)

    num_files = len(b1_files)
    array_size = num_files - 1  # 0-indexed

    print(f"Found {num_files} b1_vap files to process")
    print(f"Array indices: 0-{array_size}")

    # Build sbatch arguments
    sbatch_args = [
        f"--export=LOCATION={location_key},SKIP_EXISTING={int(skip_existing)}",
        f"--array=0-{array_size}",
        f"--output={location_key}_compress_b1_%A_%a.out",
        f"--job-name={location_key}_compress_b1",
        "compress_b1_array.sbatch",
    ]

    # Submit the array job
    print("Submitting SLURM array job...")
    job_id = submit_sbatch(sbatch_args)

    print(f"✓ Array job submitted with ID: {job_id}")
    print(f"  Location: {location_key}")
    print(f"  Files: {num_files}")
    print(f"  Skip existing: {skip_existing}")
    print(f"  Array range: 0-{array_size}")
    print()
    print("Monitor job status with:")
    print(f"  squeue -j {job_id}")
    print(f"  sacct -j {job_id}")


if __name__ == "__main__":
    args = parse_args()
    dispatch_jobs(args.location, skip_existing=args.skip_existing)
    print("Done!")
