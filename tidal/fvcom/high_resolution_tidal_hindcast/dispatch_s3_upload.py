#!/usr/bin/env python3
"""
S3 Upload Dispatcher - Discover files and submit SLURM jobs for S3 uploads.

This script:
1. Discovers files based on location and data level
2. Constructs S3 destination paths
3. Displays summary and asks for confirmation
4. Submits individual SLURM jobs for each file upload

Usage:
    python dispatch_s3_upload.py <location> --data-levels <level1> [<level2> ...]

Examples:
    # Upload all b1_vap files for Cook Inlet
    python dispatch_s3_upload.py cook_inlet --data-levels b1_vap

    # Upload multiple data levels for Puget Sound
    python dispatch_s3_upload.py puget_sound --data-levels b1_vap hsds a1_std

    # Dry run to see what would be uploaded
    python dispatch_s3_upload.py cook_inlet --data-levels b1_vap --dry-run
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Import config
from config import config

# Data level configurations with SLURM time limits (in hours)
DATA_LEVEL_CONFIG = {
    "00_raw": {"time_hours": 6, "description": "Raw data"},
    # "a1_std": {"time_hours": 6, "description": "Standardized data"},
    # "a2_std_partition": {"time_hours": 6, "description": "Standardized partition"},
    # "b1_vap": {"time_hours": 6, "description": "Value-added products"},
    # "b2_monthly_mean_vap": {
    #     "time_hours": 6,
    #     "description": "Monthly mean VAP",
    # },
    # "b3_yearly_mean_vap": {"time_hours": 6, "description": "Yearly mean VAP"},
    # "b4_vap_partition": {"time_hours": 6, "description": "VAP partition"},
    # "b5_vap_summary_parquet": {
    #     "time_hours": 6,
    #     "description": "VAP summary parquet",
    # },
    # "b6_vap_atlas_summary_parquet": {
    #     "time_hours": 6,
    #     "description": "VAP atlas summary parquet",
    # },
    # "hsds": {"time_hours": 24, "description": "HSDS format"},
}

# Location key to output_name mapping from config
LOCATION_MAP = {
    key: spec["output_name"] for key, spec in config["location_specification"].items()
}


def get_file_size_human(size_bytes):
    """Convert bytes to human-readable format."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def discover_files(location_key, data_levels):
    """
    Discover files for given location and data levels.

    Args:
        location_key: Location key (e.g., 'cook_inlet')
        data_levels: List of data level directory names (e.g., ['b1_vap', 'hsds'])

    Returns:
        Dictionary mapping data_level to list of (local_path, relative_path) tuples
    """
    # Validate location
    if location_key not in LOCATION_MAP:
        print(f"Error: Unknown location '{location_key}'", file=sys.stderr)
        print(f"Available locations: {', '.join(LOCATION_MAP.keys())}", file=sys.stderr)
        sys.exit(1)

    # Validate data levels
    for level in data_levels:
        if level not in DATA_LEVEL_CONFIG:
            print(f"Error: Unknown data level '{level}'", file=sys.stderr)
            print(
                f"Available data levels: {', '.join(DATA_LEVEL_CONFIG.keys())}",
                file=sys.stderr,
            )
            sys.exit(1)

    output_name = LOCATION_MAP[location_key]
    base_dir = Path(config["dir"]["base"])

    files_by_level = {}

    for data_level in data_levels:
        # Construct local directory path
        local_dir = base_dir / output_name / data_level

        if not local_dir.exists():
            print(f"Warning: Directory does not exist: {local_dir}", file=sys.stderr)
            files_by_level[data_level] = []
            continue

        # Discover all files recursively
        files = []
        for file_path in local_dir.rglob("*"):
            if file_path.is_file():
                # Calculate relative path from data_level directory
                relative_path = file_path.relative_to(local_dir)
                files.append((str(file_path), str(relative_path)))

        files_by_level[data_level] = sorted(files)
        print(f"Found {len(files)} files in {data_level}/")

    return files_by_level


def calculate_total_size(files_by_level):
    """Calculate total size of all files."""
    total_bytes = 0
    for files in files_by_level.values():
        for local_path, _ in files:
            if os.path.exists(local_path):
                total_bytes += os.path.getsize(local_path)
    return total_bytes


def construct_s3_destination(location_key, data_level, relative_path):
    """
    Construct S3 destination path.

    Format: <location>/<version>/<data_level>/<relative_path>

    Args:
        location_key: Location key (e.g., 'cook_inlet')
        data_level: Data level directory name (e.g., 'b1_vap')
        relative_path: Relative path within data level (e.g., '2005/file.nc')

    Returns:
        S3 destination path
    """
    output_name = LOCATION_MAP[location_key]
    version = config["dataset"]["version"]
    return f"{output_name}/{version}/{data_level}/{relative_path}"


def display_summary(location_key, files_by_level):
    """Display summary of files to be uploaded."""
    output_name = LOCATION_MAP[location_key]
    version = config["dataset"]["version"]
    total_files = sum(len(files) for files in files_by_level.values())
    total_size = calculate_total_size(files_by_level)

    print("\n" + "=" * 80)
    print("S3 UPLOAD SUMMARY")
    print("=" * 80)
    print(f"Location:      {location_key} ({output_name})")
    print(f"Version:       {version}")
    print(f"Total files:   {total_files}")
    print(f"Total size:    {get_file_size_human(total_size)}")
    print("S3 bucket:     oedi-data-drop/us-tidal")
    print("\nData Levels:")

    for data_level, files in files_by_level.items():
        level_size = sum(
            os.path.getsize(local_path)
            for local_path, _ in files
            if os.path.exists(local_path)
        )
        config_info = DATA_LEVEL_CONFIG[data_level]
        print(
            f"  {data_level:30s} {len(files):6d} files  "
            f"{get_file_size_human(level_size):>12s}  "
            f"(time limit: {config_info['time_hours']}h)"
        )

    # Show sample file paths (up to 10)
    print("\nSample S3 destinations (up to 10):")
    sample_count = 0
    for data_level, files in files_by_level.items():
        for local_path, relative_path in files[:10]:
            if sample_count >= 10:
                break
            s3_dest = construct_s3_destination(location_key, data_level, relative_path)
            print(f"  s3://oedi-data-drop/us-tidal/{s3_dest}")
            sample_count += 1
        if sample_count >= 10:
            break

    if total_files > 10:
        print(f"  ... and {total_files - 10} more files")

    print("=" * 80)


def confirm_upload():
    """Prompt user for confirmation."""
    while True:
        response = input("\nProceed with upload? [y/N]: ").strip().lower()
        if response in ["y", "yes"]:
            return True
        elif response in ["n", "no", ""]:
            return False
        else:
            print("Please enter 'y' or 'n'")


def submit_slurm_job(
    local_file,
    s3_destination,
    data_level,
    location_key,
    skip_if_exists=True,
    verify_checksum=True,
    dry_run=False,
):
    """
    Submit SLURM job for single file upload.

    Args:
        local_file: Full path to local file
        s3_destination: S3 destination path
        data_level: Data level (for time limit configuration)
        location_key: Location key (for job naming)
        skip_if_exists: Skip upload if file exists
        verify_checksum: Verify upload with checksum
        dry_run: Print command without submitting

    Returns:
        Job ID if submitted, None if dry run
    """
    time_hours = DATA_LEVEL_CONFIG[data_level]["time_hours"]
    time_minutes = time_hours * 60

    # Construct safe filename for output log
    safe_filename = Path(local_file).name.replace("/", "_")
    output_file = f"s3_upload_{safe_filename}_%j.out"

    # Create common job name for easy cancellation: s3_upload_<location>_<data_level>
    job_name = f"s3_upload_{location_key}_{data_level}"

    # Build sbatch command
    export_vars = [
        f"LOCAL_FILE={local_file}",
        f"S3_DESTINATION={s3_destination}",
        f"SKIP_IF_EXISTS={'true' if skip_if_exists else 'false'}",
        f"VERIFY_CHECKSUM={'true' if verify_checksum else 'false'}",
    ]

    sbatch_args = [
        "sbatch",
        "--parsable",
        f"--job-name={job_name}",
        f"--export={','.join(export_vars)}",
        f"--output={output_file}",
        f"--time={time_minutes}",
        "s3_upload.sbatch",
    ]

    if dry_run:
        print(f"[DRY RUN] Would submit: {' '.join(sbatch_args)}")
        return None

    try:
        result = subprocess.run(sbatch_args, capture_output=True, text=True, check=True)
        job_id = result.stdout.strip()
        return job_id
    except subprocess.CalledProcessError as e:
        print(f"Error submitting job: {e}", file=sys.stderr)
        print(f"stderr: {e.stderr}", file=sys.stderr)
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Discover files and dispatch S3 upload jobs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available Locations:
  aleutian_islands, cook_inlet, piscataqua_river, puget_sound, western_passage

Available Data Levels:
  00_raw                      - Raw data (6h time limit)
  a1_std                      - Standardized data (6h time limit)
  a2_std_partition            - Standardized partition (6h time limit)
  b1_vap                      - Value-added products (6h time limit)
  b2_monthly_mean_vap         - Monthly mean VAP (6h time limit)
  b3_yearly_mean_vap          - Yearly mean VAP (6h time limit)
  b4_vap_partition            - VAP partition (6h time limit)
  b5_vap_summary_parquet      - VAP summary parquet (6h time limit)
  b6_vap_atlas_summary_parquet - VAP atlas summary parquet (6h time limit)
  hsds                        - HSDS format (24h time limit)

Examples:
  # Upload all b1_vap files for Cook Inlet
  python dispatch_s3_upload.py cook_inlet --data-levels b1_vap

  # Upload multiple data levels
  python dispatch_s3_upload.py puget_sound --data-levels b1_vap hsds

  # Dry run (don't actually submit jobs)
  python dispatch_s3_upload.py cook_inlet --data-levels b1_vap --dry-run

  # Skip checksum verification (faster)
  python dispatch_s3_upload.py cook_inlet --data-levels b1_vap --no-verify-checksum

  # Force upload (overwrite existing files)
  python dispatch_s3_upload.py cook_inlet --data-levels b1_vap --no-skip-if-exists
        """,
    )

    parser.add_argument(
        "location",
        help="Location key (e.g., cook_inlet, puget_sound)",
    )

    parser.add_argument(
        "--data-levels",
        nargs="+",
        required=True,
        help="Data level directory names (e.g., b1_vap hsds)",
    )

    parser.add_argument(
        "--skip-if-exists",
        dest="skip_if_exists",
        action="store_true",
        default=True,
        help="Skip upload if file exists in S3 (default: True)",
    )

    parser.add_argument(
        "--no-skip-if-exists",
        dest="skip_if_exists",
        action="store_false",
        help="Upload even if file exists in S3",
    )

    parser.add_argument(
        "--verify-checksum",
        dest="verify_checksum",
        action="store_true",
        default=True,
        help="Verify upload with MD5 checksum (default: True)",
    )

    parser.add_argument(
        "--no-verify-checksum",
        dest="verify_checksum",
        action="store_false",
        help="Skip checksum verification",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions without submitting SLURM jobs",
    )

    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Skip confirmation prompt",
    )

    args = parser.parse_args()

    # Discover files
    print(f"Discovering files for location: {args.location}")
    print(f"Data levels: {', '.join(args.data_levels)}")
    print()

    files_by_level = discover_files(args.location, args.data_levels)

    # Check if any files found
    total_files = sum(len(files) for files in files_by_level.values())
    if total_files == 0:
        print("No files found to upload.", file=sys.stderr)
        sys.exit(1)

    # Display summary
    display_summary(args.location, files_by_level)

    # Confirm upload (unless --yes or --dry-run)
    if not args.yes and not args.dry_run:
        if not confirm_upload():
            print("Upload cancelled.")
            sys.exit(0)

    # Submit jobs
    print("\nSubmitting SLURM jobs...")
    submitted_jobs = []
    job_names = set()

    for data_level, files in files_by_level.items():
        print(f"\nSubmitting jobs for {data_level}...")
        job_name = f"s3_upload_{args.location}_{data_level}"
        job_names.add(job_name)

        for local_path, relative_path in files:
            s3_destination = construct_s3_destination(
                args.location, data_level, relative_path
            )

            job_id = submit_slurm_job(
                local_file=local_path,
                s3_destination=s3_destination,
                data_level=data_level,
                location_key=args.location,
                skip_if_exists=args.skip_if_exists,
                verify_checksum=args.verify_checksum,
                dry_run=args.dry_run,
            )

            if job_id:
                submitted_jobs.append(job_id)
                print(f"  Submitted job {job_id}: {Path(local_path).name}")

    # Summary
    print("\n" + "=" * 80)
    if args.dry_run:
        print(f"DRY RUN COMPLETE - Would have submitted {total_files} jobs")
    else:
        print(f"Successfully submitted {len(submitted_jobs)} SLURM jobs")
        print("\nJob Management Commands:")
        print("  Monitor all jobs:       squeue -u $USER")
        print(
            f"  Monitor by location:    squeue -u $USER -n s3_upload_{args.location}_*"
        )
        for job_name in sorted(job_names):
            print(
                f"  Monitor {job_name.split('_')[-1]:6s} jobs:    squeue -u $USER -n {job_name}"
            )
        print("\n  Cancel all upload jobs: scancel -u $USER -n 's3_upload_*'")
        for job_name in sorted(job_names):
            print(
                f"  Cancel {job_name.split('_')[-1]:6s} jobs:     scancel -u $USER -n {job_name}"
            )
    print("=" * 80)


if __name__ == "__main__":
    main()
