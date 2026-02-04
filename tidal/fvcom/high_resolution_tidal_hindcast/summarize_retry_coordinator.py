#!/usr/bin/env python3
"""
Retry coordinator for summarization jobs.

This script runs after the initial shared partition jobs complete.
It checks for missing batch files and submits retry jobs to the
standard partition for any failures.

Usage:
    python summarize_retry_coordinator.py <location>

Example:
    python summarize_retry_coordinator.py cook_inlet
"""

import argparse
import math
import re
import subprocess
import sys
import time

from config import config
from src import file_manager
from dispatch_summarize_jobs import LOCATIONS, BATCH_SIZE_MAP


def find_missing_batches(output_dir, total_faces, batch_size):
    """
    Find which batch indices are missing from the output directory.

    Args:
        output_dir: Path to the batch output directory
        total_faces: Total number of faces for this location
        batch_size: Size of each batch

    Returns:
        List of missing batch indices
    """
    total_batches = math.ceil(total_faces / batch_size)
    expected_indices = set(range(total_batches))

    # Find existing batch files
    existing_files = list(output_dir.glob("*.nc"))
    found_indices = set()

    for filepath in existing_files:
        # Extract batch index from filename pattern like "batch_5_faces_50000_59999"
        match = re.search(r"batch_(\d+)_faces_\d+_\d+", filepath.name)
        if match:
            batch_idx = int(match.group(1))
            found_indices.add(batch_idx)

    missing = sorted(expected_indices - found_indices)

    print(f"Total expected batches: {total_batches}")
    print(f"Found batches: {len(found_indices)}")
    print(f"Missing batches: {len(missing)}")

    return missing


def submit_retry_jobs(location, missing_indices, faces, batch_size, runtime_hours):
    """
    Submit retry jobs to standard partition for missing batches.

    Uses sbatch --wait to block until all jobs complete.

    Args:
        location: Location name
        missing_indices: List of batch indices to retry
        faces: Total faces
        batch_size: Batch size
        runtime_hours: Runtime limit in hours

    Returns:
        True if submission successful, False otherwise
    """
    if not missing_indices:
        print("No missing batches to retry.")
        return True

    # Format array indices for SLURM
    array_spec = format_array_indices(missing_indices)

    print(f"Submitting {len(missing_indices)} retry jobs to standard partition...")
    print(f"Array specification: {array_spec}")

    # Submit with --wait to block until complete
    retry_args = [
        "sbatch",
        "--wait",  # Block until jobs complete
        "--parsable",
        f"--export=LOCATION={location},FACES={faces},BATCH_SIZE={batch_size}",
        f"--array={array_spec}",
        f"--output={location}_retry_%A_%a.out",
        f"--job-name={location}_retry",
        f"--time={runtime_hours * 60}",
        "--partition=standard",  # Use standard partition for retries
        "summarize_single_location_batch.sbatch",
    ]

    try:
        print(f"Running: {' '.join(retry_args)}")
        result = subprocess.run(retry_args, capture_output=True, text=True, check=True)
        print(f"Retry jobs completed. Job ID: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Retry job submission failed: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return False


def format_array_indices(indices):
    """
    Format a list of indices into SLURM array specification.

    Converts [1, 2, 3, 5, 7, 8, 9] to "1-3,5,7-9"

    Args:
        indices: List of integer indices

    Returns:
        SLURM array specification string
    """
    if not indices:
        return ""

    indices = sorted(indices)
    ranges = []
    start = indices[0]
    end = indices[0]

    for i in indices[1:]:
        if i == end + 1:
            end = i
        else:
            if start == end:
                ranges.append(str(start))
            else:
                ranges.append(f"{start}-{end}")
            start = i
            end = i

    # Don't forget the last range
    if start == end:
        ranges.append(str(start))
    else:
        ranges.append(f"{start}-{end}")

    return ",".join(ranges)


def main():
    parser = argparse.ArgumentParser(
        description="Retry coordinator for failed summarization jobs"
    )
    parser.add_argument(
        "location", help="Location to process (e.g., cook_inlet, puget_sound)"
    )
    args = parser.parse_args()

    location = args.location

    if location not in LOCATIONS:
        print(f"Error: Unknown location '{location}'")
        print(f"Valid locations: {', '.join(LOCATIONS.keys())}")
        sys.exit(1)

    loc_config = LOCATIONS[location]
    faces = loc_config["faces"]
    temporal_resolution = loc_config["temporal_resolution"]
    batch_size = BATCH_SIZE_MAP[temporal_resolution]
    # Use longer runtime for retries (dedicated resources should be faster, but give buffer)
    runtime_hours = loc_config.get(
        "retry_runtime_hours", loc_config["process_runtime_hours"] + 2
    )

    print(f"=== Retry Coordinator for {location} ===")
    print(f"Faces: {faces}, Batch size: {batch_size}")

    # Get location config for file_manager
    location_spec = config["location_specification"].get(location)

    if not location_spec:
        # Try alternate key formats
        for key, spec in config["location_specification"].items():
            if location in key or location in spec.get("output_name", "").lower():
                location_spec = spec
                break

    if not location_spec:
        print(f"Error: Could not find location '{location}' in config")
        sys.exit(1)

    # Get output directory
    output_dir = file_manager.get_yearly_summary_by_face_vap_output_dir(
        config, location_spec
    )
    print(f"Output directory: {output_dir}")

    # Check for missing batches
    missing = find_missing_batches(output_dir, faces, batch_size)

    if not missing:
        print("All batches completed successfully!")
        sys.exit(0)

    print(f"Missing batch indices: {missing}")

    # Submit retry jobs
    success = submit_retry_jobs(location, missing, faces, batch_size, runtime_hours)

    if not success:
        print("Retry job submission failed!")
        sys.exit(1)

    # Wait a moment for filesystem to sync
    time.sleep(5)

    # Verify all batches now exist
    still_missing = find_missing_batches(output_dir, faces, batch_size)

    if still_missing:
        print(f"ERROR: {len(still_missing)} batches still missing after retry!")
        print(f"Missing indices: {still_missing}")
        sys.exit(1)

    print("All batches completed successfully after retry!")
    sys.exit(0)


if __name__ == "__main__":
    main()
