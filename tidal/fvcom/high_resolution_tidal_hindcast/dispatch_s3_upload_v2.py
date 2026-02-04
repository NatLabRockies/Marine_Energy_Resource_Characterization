#!/usr/bin/env python3
"""
S3 Upload Dispatcher v2 - Manifest-based SLURM array job submission.

This version creates SQLite manifests and submits SLURM array jobs for
parallel uploads with automatic retry and failure tracking.
"""

import argparse
import math
import sqlite3
import subprocess
import sys
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

from config import config
from create_s3_manifest import create_manifest
from upload_to_s3 import S3_BASE_PATH, S3_BUCKET, S3_PROFILE

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
        "time_hours": 1,
        "description": "Monthly mean VAP",
    },
    "b3_yearly_mean_vap": {"time_hours": 1, "description": "Yearly mean VAP"},
    "b1_vap_by_point_partition": {"time_hours": 6, "description": "VAP partition"},
    "b4_vap_summary_parquet": {
        "time_hours": 1,
        "description": "VAP summary parquet",
    },
    "b5_vap_atlas_summary_parquet": {
        "time_hours": 1,
        "description": "VAP atlas summary parquet",
    },
    "hsds": {"time_hours": 24, "description": "HSDS format"},
    "manifest": {"time_hours": 1, "description": "Global manifest (JSON)"},
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


def list_existing_s3_files(location_key, data_level):
    """
    List existing files on S3 for a given location and data level.

    Uses S3 list_objects_v2 with pagination to get all object keys.
    Only extracts filenames (not full paths) for fast comparison.

    Args:
        location_key: Location key (e.g., 'cook_inlet')
        data_level: Data level (e.g., 'b1_vap')

    Returns:
        Set of filenames that exist on S3
    """
    output_name = LOCATION_MAP[location_key]
    version = config["dataset"]["version"]

    # Build S3 prefix to list
    # Format: us-tidal/<output_name>/v<version>/<data_level>/
    # Note: version has 'v' prefix to match file_manager convention
    s3_prefix = f"{S3_BASE_PATH}/{output_name}/v{version}/{data_level}/"

    print(f"   Listing S3 objects with prefix: s3://{S3_BUCKET}/{s3_prefix}")

    try:
        session = boto3.Session(profile_name=S3_PROFILE)
        s3_client = session.client("s3")
    except Exception as e:
        print(f"   Warning: Could not create S3 client: {e}", file=sys.stderr)
        return set()

    existing_files = set()
    continuation_token = None

    try:
        while True:
            # Build request parameters
            list_kwargs = {
                "Bucket": S3_BUCKET,
                "Prefix": s3_prefix,
                "MaxKeys": 1000,
            }
            if continuation_token:
                list_kwargs["ContinuationToken"] = continuation_token

            response = s3_client.list_objects_v2(**list_kwargs)

            # Extract filenames from keys
            if "Contents" in response:
                for obj in response["Contents"]:
                    # Extract just the filename from the full key
                    filename = obj["Key"].split("/")[-1]
                    if filename:  # Skip empty strings (directory markers)
                        existing_files.add(filename)

            # Check if there are more results
            if response.get("IsTruncated"):
                continuation_token = response.get("NextContinuationToken")
            else:
                break

        print(f"   Found {len(existing_files)} existing files on S3")
        return existing_files

    except ClientError as e:
        print(f"   Warning: Could not list S3 objects: {e}", file=sys.stderr)
        return set()


def list_existing_s3_manifest_files():
    """
    List existing manifest files on S3.

    Returns:
        Set of filenames that exist on S3
    """
    manifest_version = config["manifest"]["version"]

    # Build S3 prefix for manifest: us-tidal/manifest/v{manifest_version}/
    s3_prefix = f"{S3_BASE_PATH}/manifest/v{manifest_version}/"

    print(f"   Listing S3 objects with prefix: s3://{S3_BUCKET}/{s3_prefix}")

    try:
        session = boto3.Session(profile_name=S3_PROFILE)
        s3_client = session.client("s3")
    except Exception as e:
        print(f"   Warning: Could not create S3 client: {e}", file=sys.stderr)
        return set()

    existing_files = set()
    continuation_token = None

    try:
        while True:
            list_kwargs = {
                "Bucket": S3_BUCKET,
                "Prefix": s3_prefix,
                "MaxKeys": 1000,
            }
            if continuation_token:
                list_kwargs["ContinuationToken"] = continuation_token

            response = s3_client.list_objects_v2(**list_kwargs)

            if "Contents" in response:
                for obj in response["Contents"]:
                    filename = obj["Key"].split("/")[-1]
                    if filename:
                        existing_files.add(filename)

            if response.get("IsTruncated"):
                continuation_token = response.get("NextContinuationToken")
            else:
                break

        print(f"   Found {len(existing_files)} existing files on S3")
        return existing_files

    except ClientError as e:
        print(f"   Warning: Could not list S3 objects: {e}", file=sys.stderr)
        return set()


def mark_existing_files_as_skipped(manifest_path, existing_s3_files):
    """
    Mark files in manifest as 'skipped' if they already exist on S3.

    Compares filenames only (not full paths) for speed.

    Args:
        manifest_path: Path to SQLite manifest
        existing_s3_files: Set of filenames that exist on S3

    Returns:
        Number of files marked as skipped
    """
    conn = sqlite3.connect(manifest_path)
    cursor = conn.cursor()

    # Get all pending files from manifest
    cursor.execute(
        """
        SELECT file_index, s3_destination
        FROM files
        WHERE upload_status = 'pending'
        """
    )

    files_to_skip = []
    for file_index, s3_destination in cursor.fetchall():
        # Extract filename from s3_destination
        filename = s3_destination.split("/")[-1]
        if filename in existing_s3_files:
            files_to_skip.append(file_index)

    # Mark files as skipped
    if files_to_skip:
        cursor.executemany(
            """
            UPDATE files
            SET upload_status = 'skipped'
            WHERE file_index = ?
            """,
            [(idx,) for idx in files_to_skip],
        )
        conn.commit()

    conn.close()
    return len(files_to_skip)


def count_pending_manifest_files(manifest_path):
    """Count pending files in manifest (excludes skipped)."""
    conn = sqlite3.connect(manifest_path)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM files WHERE upload_status = 'pending'")
    count = cursor.fetchone()[0]
    conn.close()
    return count


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
  {", ".join(LOCATION_MAP.keys())}, manifest

Available Data Levels:
  00_raw, a1_std, a2_std_partition, b1_vap, b1_vap_daily_compressed,
  b2_monthly_mean_vap, b3_yearly_mean_vap, b1_vap_by_point_partition,
  b4_vap_summary_parquet, b5_vap_atlas_summary_parquet, hsds

Examples:
  # Upload all b1_vap files for Cook Inlet
  python dispatch_s3_upload_v2.py cook_inlet --data-levels b1_vap

  # Upload multiple data levels
  python dispatch_s3_upload_v2.py puget_sound --data-levels b1_vap hsds

  # Upload global manifest (no --data-levels needed)
  python dispatch_s3_upload_v2.py manifest

  # Dry run
  python dispatch_s3_upload_v2.py cook_inlet --data-levels b1_vap --dry-run

  # Skip checksum verification (faster)
  python dispatch_s3_upload_v2.py cook_inlet --data-levels b1_vap --no-verify-checksum

  # Skip files that already exist on S3
  python dispatch_s3_upload_v2.py cook_inlet --data-levels b1_vap --skip-if-uploaded
        """,
    )

    parser.add_argument(
        "location",
        help="Location key (e.g., cook_inlet, puget_sound) or 'manifest' for global manifest upload",
    )

    parser.add_argument(
        "--data-levels",
        nargs="+",
        help="Data level directory names (e.g., b1_vap hsds). Not required for 'manifest' location.",
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

    parser.add_argument(
        "--skip-if-uploaded",
        action="store_true",
        help="Skip files that already exist on S3 (filename check only)",
    )

    args = parser.parse_args()

    # Check if this is a manifest upload (special case)
    is_manifest_upload = args.location == "manifest"

    # Validate location and data-levels
    if is_manifest_upload:
        # Manifest upload: data-levels is ignored
        if args.data_levels:
            print(
                "Note: --data-levels is ignored for manifest uploads",
                file=sys.stderr,
            )
        data_levels = ["manifest"]
    else:
        # Location-specific upload: validate location and require data-levels
        if args.location not in LOCATION_MAP:
            print(f"Error: Unknown location '{args.location}'", file=sys.stderr)
            print(
                f"Available locations: {', '.join(LOCATION_MAP.keys())}, manifest",
                file=sys.stderr,
            )
            sys.exit(1)

        if not args.data_levels:
            print(
                "Error: --data-levels is required for location-specific uploads",
                file=sys.stderr,
            )
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

        data_levels = args.data_levels

    # Create log directory
    create_log_directory()

    print("=" * 80)
    print("S3 UPLOAD DISPATCHER v2 - Manifest-Based Upload")
    print("=" * 80)
    if is_manifest_upload:
        manifest_version = config["manifest"]["version"]
        print("Mode:          Global manifest upload")
        print(f"Manifest ver:  {manifest_version}")
    else:
        print(f"Location:      {args.location} ({LOCATION_MAP[args.location]})")
        print(f"Data levels:   {', '.join(data_levels)}")
    print(f"Max jobs:      {MAX_SLURM_JOBS}")
    if args.skip_if_uploaded:
        print("Skip existing: Enabled (filename check)")
    print("=" * 80)

    # Process each data level
    all_job_ids = []

    for data_level in data_levels:
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

        # Skip files that already exist on S3 if requested
        skipped_files = 0
        if args.skip_if_uploaded:
            print("\n2. Checking for existing files on S3...")
            if is_manifest_upload:
                existing_files = list_existing_s3_manifest_files()
            else:
                existing_files = list_existing_s3_files(args.location, data_level)
            if existing_files:
                skipped_files = mark_existing_files_as_skipped(
                    manifest_path, existing_files
                )
                print(f"   Skipped {skipped_files} files (already on S3)")

        # Count pending files (after skipping)
        pending_files = count_pending_manifest_files(manifest_path)

        # Calculate batch parameters based on pending files
        if pending_files == 0:
            print("\n   No files to upload (all skipped or none found)")
            continue

        num_jobs, files_per_job = calculate_batch_params(pending_files)

        step_num = 3 if args.skip_if_uploaded else 2
        print(f"\n{step_num}. Upload Job Configuration:")
        print(f"   Total files:     {total_files}")
        if skipped_files > 0:
            print(f"   Skipped files:   {skipped_files}")
            print(f"   Pending files:   {pending_files}")
        print(f"   Files per job:   {files_per_job}")
        print(f"   Number of jobs:  {num_jobs}")
        print(f"   Time limit:      {DATA_LEVEL_CONFIG[data_level]['time_hours']}h")

        # Submit upload jobs
        step_num += 1
        print(f"\n{step_num}. Submitting upload jobs...")
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
            step_num += 1
            print(f"\n{step_num}. Submitting failure manifest job...")
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
