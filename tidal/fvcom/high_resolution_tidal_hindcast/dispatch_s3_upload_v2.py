#!/usr/bin/env python3
"""
S3 Upload Dispatcher v2 - Manifest-based SLURM array job submission.

This version creates SQLite manifests and submits SLURM array jobs for
parallel uploads with automatic retry and failure tracking.
"""

import argparse
import math
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

from config import config
from create_s3_manifest import create_manifest

# Maximum SLURM array jobs (global limit)
MAX_SLURM_JOBS = 500

# Data level configurations with SLURM time limits (in hours)
DATA_LEVEL_CONFIG = {
    "00_raw": {"time_hours": 6, "description": "Raw data"},
    "a1_std": {"time_hours": 6, "description": "Standardized data"},
    "a2_std_partition": {"time_hours": 6, "description": "Standardized partition"},
    "b1_vap": {"time_hours": 6, "description": "Value-added products"},
    "b1_vap_daily_compressed": {
        "time_hours": 6,
        "description": "Compressed VAP (daily)",
    },
    "b2_monthly_mean_vap": {
        "time_hours": 6,
        "description": "Monthly mean VAP",
    },
    "b3_yearly_mean_vap": {"time_hours": 6, "description": "Yearly mean VAP"},
    "b4_vap_partition": {"time_hours": 6, "description": "VAP partition"},
    "b5_vap_summary_parquet": {
        "time_hours": 6,
        "description": "VAP summary parquet",
    },
    "b6_vap_atlas_summary_parquet": {
        "time_hours": 6,
        "description": "VAP atlas summary parquet",
    },
    "hsds": {"time_hours": 24, "description": "HSDS format"},
}

# Location key to output_name mapping from config
LOCATION_MAP = {
    key: spec["output_name"] for key, spec in config["location_specification"].items()
}


def create_log_directory():
    """Create directory for SLURM logs."""
    log_dir = Path("cache/s3_upload/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def count_manifest_files(manifest_path):
    """Count total files in manifest."""
    conn = sqlite3.connect(manifest_path)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM files")
    count = cursor.fetchone()[0]
    conn.close()
    return count


def calculate_batch_params(total_files):
    """
    Calculate optimal batching parameters.

    Args:
        total_files: Total number of files

    Returns:
        (num_jobs, files_per_job) tuple
    """
    if total_files <= MAX_SLURM_JOBS:
        # Each job processes 1 file
        return total_files, 1
    else:
        # Batch files to stay under MAX_SLURM_JOBS
        files_per_job = math.ceil(total_files / MAX_SLURM_JOBS)
        num_jobs = math.ceil(total_files / files_per_job)
        return num_jobs, files_per_job


def submit_upload_jobs(
    manifest_path,
    location_key,
    data_level,
    num_jobs,
    files_per_job,
    verify_checksum=True,
    dry_run=False,
):
    """
    Submit SLURM array jobs for uploads.

    Args:
        manifest_path: Path to SQLite manifest
        location_key: Location key
        data_level: Data level
        num_jobs: Number of array jobs
        files_per_job: Files per job
        verify_checksum: Enable checksum verification
        dry_run: Don't actually submit

    Returns:
        Job ID if submitted, None if dry run
    """
    time_hours = DATA_LEVEL_CONFIG[data_level]["time_hours"]

    # Format time as HH:MM:SS for SLURM
    time_str = f"{time_hours}:00:00"

    # Build environment variables
    export_vars = [
        f"MANIFEST_PATH={manifest_path}",
        f"FILES_PER_JOB={files_per_job}",
        f"VERIFY_CHECKSUM={'true' if verify_checksum else 'false'}",
    ]

    # Build sbatch command
    sbatch_args = [
        "sbatch",
        "--parsable",
        f"--array=0-{num_jobs - 1}",
        f"--export={','.join(export_vars)}",
        f"--time={time_str}",
        "s3_upload_array.sbatch",
    ]

    if dry_run:
        print(f"\n[DRY RUN] Would submit: {' '.join(sbatch_args)}")
        return None

    try:
        result = subprocess.run(sbatch_args, capture_output=True, text=True, check=True)
        job_id = result.stdout.strip()
        return job_id
    except subprocess.CalledProcessError as e:
        print(f"Error submitting upload jobs: {e}", file=sys.stderr)
        print(f"stderr: {e.stderr}", file=sys.stderr)
        return None


def submit_failure_manifest_job(manifest_path, dependency_job_id, dry_run=False):
    """
    Submit failure manifest creation job with dependency.

    Args:
        manifest_path: Path to SQLite manifest
        dependency_job_id: Job ID to depend on
        dry_run: Don't actually submit

    Returns:
        Job ID if submitted, None if dry run
    """
    export_vars = [f"MANIFEST_PATH={manifest_path}"]

    sbatch_args = [
        "sbatch",
        "--parsable",
        f"--dependency=afterany:{dependency_job_id}",
        f"--export={','.join(export_vars)}",
        "s3_create_failure_manifest.sbatch",
    ]

    if dry_run:
        print(f"[DRY RUN] Would submit: {' '.join(sbatch_args)}")
        return None

    try:
        result = subprocess.run(sbatch_args, capture_output=True, text=True, check=True)
        job_id = result.stdout.strip()
        return job_id
    except subprocess.CalledProcessError as e:
        print(f"Error submitting failure job: {e}", file=sys.stderr)
        print(f"stderr: {e.stderr}", file=sys.stderr)
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Dispatch S3 upload jobs using manifest-based SLURM arrays",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available Locations:
  {", ".join(LOCATION_MAP.keys())}

Available Data Levels:
  00_raw, a1_std, a2_std_partition, b1_vap, b1_vap_daily_compressed,
  b2_monthly_mean_vap, b3_yearly_mean_vap, b4_vap_partition,
  b5_vap_summary_parquet, b6_vap_atlas_summary_parquet, hsds

Examples:
  # Upload all b1_vap files for Cook Inlet
  python dispatch_s3_upload_v2.py cook_inlet --data-levels b1_vap

  # Upload multiple data levels
  python dispatch_s3_upload_v2.py puget_sound --data-levels b1_vap hsds

  # Dry run
  python dispatch_s3_upload_v2.py cook_inlet --data-levels b1_vap --dry-run

  # Skip checksum verification (faster)
  python dispatch_s3_upload_v2.py cook_inlet --data-levels b1_vap --no-verify-checksum
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
        "--verify-checksum",
        dest="verify_checksum",
        action="store_true",
        default=True,
        help="Verify upload with checksum (default: True)",
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

    args = parser.parse_args()

    # Validate location
    if args.location not in LOCATION_MAP:
        print(f"Error: Unknown location '{args.location}'", file=sys.stderr)
        print(f"Available locations: {', '.join(LOCATION_MAP.keys())}", file=sys.stderr)
        sys.exit(1)

    # Validate data levels
    for level in args.data_levels:
        if level not in DATA_LEVEL_CONFIG:
            print(f"Error: Unknown data level '{level}'", file=sys.stderr)
            print(
                f"Available data levels: {', '.join(DATA_LEVEL_CONFIG.keys())}",
                file=sys.stderr,
            )
            sys.exit(1)

    # Create log directory
    create_log_directory()

    print("=" * 80)
    print("S3 UPLOAD DISPATCHER v2 - Manifest-Based Upload")
    print("=" * 80)
    print(f"Location:      {args.location} ({LOCATION_MAP[args.location]})")
    print(f"Data levels:   {', '.join(args.data_levels)}")
    print(f"Max jobs:      {MAX_SLURM_JOBS}")
    print("=" * 80)

    # Process each data level
    all_job_ids = []

    for data_level in args.data_levels:
        print(f"\n{'=' * 80}")
        print(f"Processing: {data_level}")
        print(f"{'=' * 80}")

        # Create manifest (always run to get accurate file counts)
        print("\n1. Creating manifest...")

        try:
            manifest_path = create_manifest(args.location, data_level)
            print(f"   Manifest created: {manifest_path}")
        except Exception as e:
            print(f"Error creating manifest: {e}", file=sys.stderr)
            continue

        # Count files from manifest
        total_files = count_manifest_files(manifest_path)

        # Calculate batch parameters
        num_jobs, files_per_job = calculate_batch_params(total_files)

        print("\n2. Upload Job Configuration:")
        print(f"   Total files:     {total_files}")
        print(f"   Files per job:   {files_per_job}")
        print(f"   Number of jobs:  {num_jobs}")
        print(f"   Time limit:      {DATA_LEVEL_CONFIG[data_level]['time_hours']}h")

        # Submit upload jobs
        print("\n3. Submitting upload jobs...")
        upload_job_id = submit_upload_jobs(
            manifest_path,
            args.location,
            data_level,
            num_jobs,
            files_per_job,
            args.verify_checksum,
            args.dry_run,
        )

        if upload_job_id:
            print(f"   Upload job ID: {upload_job_id}")
            all_job_ids.append((data_level, upload_job_id))

            # Submit failure manifest job
            print("\n4. Submitting failure manifest job...")
            failure_job_id = submit_failure_manifest_job(
                manifest_path, upload_job_id, args.dry_run
            )

            if failure_job_id:
                print(f"   Failure job ID: {failure_job_id}")
        else:
            print("   Upload jobs not submitted")

    # Final summary
    print(f"\n{'=' * 80}")
    if args.dry_run:
        print("DRY RUN COMPLETE")
    else:
        print("JOBS SUBMITTED")
        print(f"\nSubmitted {len(all_job_ids)} upload job arrays:")
        for data_level, job_id in all_job_ids:
            print(f"  {data_level:30s} Job ID: {job_id}")

        print("\nMonitor jobs:")
        print("  squeue -u $USER")
        print("  squeue -j <job_id>")

        print("\nCancel all upload jobs:")
        print("  scancel -u $USER -n s3_upload")

        print("\nLogs directory:")
        print("  cache/s3_upload/logs/")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
