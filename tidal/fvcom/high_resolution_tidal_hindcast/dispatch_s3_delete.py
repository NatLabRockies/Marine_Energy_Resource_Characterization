#!/usr/bin/env python3
"""
S3 Delete Dispatcher - Delete files from S3 with parallel execution.

This script lists all files under an S3 prefix and deletes them in parallel
using either local threads or SLURM array jobs (for large deletions >10k files).

Usage:
    python dispatch_s3_delete.py <path-relative-to-base> [options]

Examples:
    # Dry run (default - shows what would be deleted)
    python dispatch_s3_delete.py AK_cook_inlet/v1.0/b1_vap

    # Actually delete with confirmation
    python dispatch_s3_delete.py AK_cook_inlet/v1.0/b1_vap --confirm

    # Force SLURM dispatch
    python dispatch_s3_delete.py AK_cook_inlet/v1.0/b1_vap --confirm --slurm

    # Custom worker count
    python dispatch_s3_delete.py AK_cook_inlet/v1.0/b1_vap --confirm --workers 32
"""

import argparse
import json
import math
import random
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

from upload_to_s3 import S3_BASE_PATH, S3_BUCKET, S3_PROFILE

# Default configuration
DEFAULT_WORKERS = 16
SLURM_THRESHOLD = 10_000  # Use SLURM for >10k files
MAX_SLURM_JOBS = 100  # Fewer concurrent jobs to avoid S3 rate limiting
FILES_PER_SLURM_JOB = 5000  # More files per job for reliability

# Rate limiting configuration
MAX_RETRIES = 8  # Max retry attempts per batch
BASE_DELAY = 1.0  # Base delay in seconds for exponential backoff
MAX_DELAY = 120.0  # Maximum delay between retries (2 minutes)
BATCH_DELAY = 0.2  # Delay between successful batches (seconds)


def get_s3_client():
    """Create and return an S3 client using the configured profile."""
    session = boto3.Session(profile_name=S3_PROFILE)
    return session.client("s3")


def list_s3_objects(s3_prefix: str) -> list[str]:
    """
    List all object keys under the given S3 prefix.

    Args:
        s3_prefix: Full S3 prefix (e.g., 'us-tidal/AK_cook_inlet/v1.0/b1_vap/')

    Returns:
        List of full S3 object keys
    """
    s3_client = get_s3_client()
    objects = []
    continuation_token = None

    print(f"Listing objects under s3://{S3_BUCKET}/{s3_prefix}")

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
                # Skip directory markers (keys ending with /)
                if not obj["Key"].endswith("/"):
                    objects.append(obj["Key"])

        if response.get("IsTruncated"):
            continuation_token = response.get("NextContinuationToken")
            # Progress indicator for large listings
            if len(objects) % 10000 == 0:
                print(f"  Found {len(objects)} objects so far...")
        else:
            break

    return objects


def delete_single_object(s3_client, key: str) -> tuple[str, bool, str]:
    """
    Delete a single S3 object.

    Args:
        s3_client: Boto3 S3 client
        key: S3 object key to delete

    Returns:
        Tuple of (key, success, error_message)
    """
    try:
        s3_client.delete_object(Bucket=S3_BUCKET, Key=key)
        return (key, True, "")
    except ClientError as e:
        return (key, False, str(e))


def is_retryable_error(error: ClientError) -> bool:
    """Check if an S3 error is retryable (rate limiting or transient)."""
    error_code = error.response.get("Error", {}).get("Code", "")
    return error_code in ("SlowDown", "ServiceUnavailable", "InternalError")


def delete_single_batch_with_retry(
    s3_client, batch: list[str]
) -> tuple[int, int, list[str]]:
    """
    Delete a single batch (up to 1000 keys) with exponential backoff retry.

    Args:
        s3_client: Boto3 S3 client
        batch: List of S3 object keys to delete (max 1000)

    Returns:
        Tuple of (success_count, error_count, error_messages)
    """
    delete_request = {"Objects": [{"Key": k} for k in batch], "Quiet": True}

    for attempt in range(MAX_RETRIES):
        try:
            response = s3_client.delete_objects(Bucket=S3_BUCKET, Delete=delete_request)

            # Quiet mode only returns errors
            if "Errors" in response:
                error_count = len(response["Errors"])
                success_count = len(batch) - error_count
                errors = [
                    f"{err['Key']}: {err['Message']}" for err in response["Errors"]
                ]
                return success_count, error_count, errors
            else:
                return len(batch), 0, []

        except ClientError as e:
            if is_retryable_error(e) and attempt < MAX_RETRIES - 1:
                # Exponential backoff with jitter
                delay = min(BASE_DELAY * (2**attempt) + random.uniform(0, 1), MAX_DELAY)
                print(
                    f"    Rate limited, retrying in {delay:.1f}s "
                    f"(attempt {attempt + 1}/{MAX_RETRIES})"
                )
                time.sleep(delay)
            else:
                # Non-retryable error or max retries exceeded
                return (
                    0,
                    len(batch),
                    [f"Batch delete failed after {attempt + 1} attempts: {e}"],
                )

    # Should not reach here, but just in case
    return 0, len(batch), ["Max retries exceeded"]


def delete_objects_batch(s3_client, keys: list[str]) -> tuple[int, int, list[str]]:
    """
    Delete a batch of S3 objects using delete_objects API with retry logic.

    Args:
        s3_client: Boto3 S3 client
        keys: List of S3 object keys to delete

    Returns:
        Tuple of (success_count, error_count, error_messages)
    """
    success_count = 0
    error_count = 0
    errors = []

    # S3 delete_objects can handle up to 1000 objects per call
    for i in range(0, len(keys), 1000):
        batch = keys[i : i + 1000]

        batch_success, batch_errors, batch_msgs = delete_single_batch_with_retry(
            s3_client, batch
        )

        success_count += batch_success
        error_count += batch_errors
        errors.extend(batch_msgs)

        # Small delay between batches to avoid overwhelming S3
        if i + 1000 < len(keys):
            time.sleep(BATCH_DELAY)

    return success_count, error_count, errors


def delete_with_threads(keys: list[str], num_workers: int) -> tuple[int, int]:
    """
    Delete S3 objects using multiple threads.

    Uses batch delete API for efficiency - each thread processes batches of 1000.

    Args:
        keys: List of S3 object keys to delete
        num_workers: Number of parallel workers

    Returns:
        Tuple of (success_count, error_count)
    """
    total_success = 0
    total_errors = 0

    # Split keys into chunks for parallel processing
    # Each chunk will use batch delete internally
    chunk_size = max(1000, len(keys) // num_workers)
    chunks = [keys[i : i + chunk_size] for i in range(0, len(keys), chunk_size)]

    print(f"\nDeleting {len(keys)} objects using {num_workers} workers...")
    print(f"Split into {len(chunks)} chunks")

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Each worker gets its own S3 client
        futures = []
        for chunk in chunks:
            s3_client = get_s3_client()
            future = executor.submit(delete_objects_batch, s3_client, chunk)
            futures.append(future)

        # Collect results
        for i, future in enumerate(as_completed(futures)):
            success, errors, error_msgs = future.result()
            total_success += success
            total_errors += errors

            # Progress update
            completed = i + 1
            print(
                f"  Progress: {completed}/{len(chunks)} chunks "
                f"({total_success} deleted, {total_errors} errors)"
            )

            # Print any errors
            for msg in error_msgs[:5]:  # Limit error output
                print(f"    Error: {msg}")
            if len(error_msgs) > 5:
                print(f"    ... and {len(error_msgs) - 5} more errors")

    return total_success, total_errors


def create_delete_manifest(keys: list[str], manifest_path: Path) -> None:
    """
    Create a manifest file for SLURM array job processing.

    Args:
        keys: List of S3 object keys to delete
        manifest_path: Path to write the manifest
    """
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    # Write keys as JSON lines for easy parsing
    with open(manifest_path, "w") as f:
        json.dump({"bucket": S3_BUCKET, "keys": keys}, f)

    print(f"Created manifest: {manifest_path}")
    print(f"  Total keys: {len(keys)}")


def calculate_slurm_params(total_files: int) -> tuple[int, int]:
    """
    Calculate SLURM array job parameters.

    Args:
        total_files: Total number of files to delete

    Returns:
        Tuple of (num_jobs, files_per_job)
    """
    files_per_job = FILES_PER_SLURM_JOB

    num_jobs = math.ceil(total_files / files_per_job)

    # Cap at MAX_SLURM_JOBS
    if num_jobs > MAX_SLURM_JOBS:
        files_per_job = math.ceil(total_files / MAX_SLURM_JOBS)
        num_jobs = math.ceil(total_files / files_per_job)

    return num_jobs, files_per_job


def submit_slurm_delete_job(
    manifest_path: Path, num_jobs: int, files_per_job: int, dry_run: bool = False
) -> str | None:
    """
    Submit SLURM array job for deletion.

    Args:
        manifest_path: Path to the manifest file
        num_jobs: Number of array tasks
        files_per_job: Files per task
        dry_run: If True, don't actually submit

    Returns:
        Job ID if submitted, None otherwise
    """
    export_vars = [
        f"DELETE_MANIFEST={manifest_path}",
        f"FILES_PER_JOB={files_per_job}",
        f"S3_BUCKET={S3_BUCKET}",
        f"S3_PROFILE={S3_PROFILE}",
    ]

    sbatch_args = [
        "sbatch",
        "--parsable",
        f"--array=0-{num_jobs - 1}",
        f"--export={','.join(export_vars)}",
        "--time=2:00:00",
        "s3_delete_array.sbatch",
    ]

    if dry_run:
        print(f"\n[DRY RUN] Would submit: {' '.join(sbatch_args)}")
        return None

    try:
        result = subprocess.run(sbatch_args, capture_output=True, text=True, check=True)
        job_id = result.stdout.strip()
        return job_id
    except subprocess.CalledProcessError as e:
        print(f"Error submitting SLURM job: {e}", file=sys.stderr)
        print(f"stderr: {e.stderr}", file=sys.stderr)
        return None
    except FileNotFoundError:
        print("Error: sbatch command not found. Are you on an HPC system?")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Delete S3 files with parallel execution",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run - list files that would be deleted
  python dispatch_s3_delete.py AK_cook_inlet/v1.0/b1_vap

  # Actually delete (requires --confirm)
  python dispatch_s3_delete.py AK_cook_inlet/v1.0/b1_vap --confirm

  # Force SLURM dispatch even for small file counts
  python dispatch_s3_delete.py AK_cook_inlet/v1.0/b1_vap --confirm --slurm

  # Use more workers for faster local deletion
  python dispatch_s3_delete.py AK_cook_inlet/v1.0/b1_vap --confirm --workers 32

Notes:
  - Paths are relative to the S3 base path (us-tidal/)
  - Full S3 path: s3://oedi-data-drop/us-tidal/<your-path>
  - For >10,000 files, SLURM dispatch is used automatically
        """,
    )

    parser.add_argument(
        "path",
        help="Path relative to S3 base path (e.g., AK_cook_inlet/v1.0/b1_vap)",
    )

    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Required to actually delete files (safety mechanism)",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Number of parallel workers for local deletion (default: {DEFAULT_WORKERS})",
    )

    parser.add_argument(
        "--slurm",
        action="store_true",
        help=f"Force SLURM dispatch (auto-enabled for >{SLURM_THRESHOLD} files)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes (implied if --confirm not set)",
    )

    args = parser.parse_args()

    # Build full S3 prefix
    # Ensure path doesn't have leading/trailing slashes, then add trailing slash
    relative_path = args.path.strip("/")
    s3_prefix = f"{S3_BASE_PATH}/{relative_path}/"

    print("=" * 80)
    print("S3 DELETE DISPATCHER")
    print("=" * 80)
    print(f"S3 Bucket:     {S3_BUCKET}")
    print(f"S3 Profile:    {S3_PROFILE}")
    print(f"S3 Prefix:     {s3_prefix}")
    print(f"Full path:     s3://{S3_BUCKET}/{s3_prefix}")
    print(f"Workers:       {args.workers}")
    print("=" * 80)

    # List objects
    print("\n1. Listing S3 objects...")
    try:
        keys = list_s3_objects(s3_prefix)
    except ClientError as e:
        print(f"Error listing S3 objects: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"\n   Found {len(keys)} objects")

    if not keys:
        print("\n   No objects found. Nothing to delete.")
        sys.exit(0)

    # Show sample of files
    print("\n   Sample files:")
    for key in keys[:5]:
        print(f"     {key}")
    if len(keys) > 5:
        print(f"     ... and {len(keys) - 5} more")

    # Determine execution mode
    use_slurm = args.slurm or len(keys) > SLURM_THRESHOLD
    is_dry_run = args.dry_run or not args.confirm

    if is_dry_run:
        print("\n" + "=" * 80)
        print("DRY RUN MODE - No files will be deleted")
        print("=" * 80)
        print(f"\nWould delete {len(keys)} objects from:")
        print(f"  s3://{S3_BUCKET}/{s3_prefix}")

        if use_slurm:
            num_jobs, files_per_job = calculate_slurm_params(len(keys))
            print("\nWould use SLURM dispatch:")
            print(f"  Array jobs: {num_jobs}")
            print(f"  Files per job: {files_per_job}")
        else:
            print("\nWould use local threaded deletion:")
            print(f"  Workers: {args.workers}")

        print("\nTo actually delete, run with --confirm flag:")
        print(f"  python dispatch_s3_delete.py {args.path} --confirm")
        sys.exit(0)

    # Confirm deletion
    print("\n" + "=" * 80)
    print("DELETION CONFIRMATION")
    print("=" * 80)
    print(f"\nAbout to DELETE {len(keys)} objects from:")
    print(f"  s3://{S3_BUCKET}/{s3_prefix}")
    print("\nThis action cannot be undone!")

    # Execute deletion
    if use_slurm:
        print("\n2. Using SLURM dispatch (large file count)...")
        num_jobs, files_per_job = calculate_slurm_params(len(keys))
        print(f"   Array jobs: {num_jobs}")
        print(f"   Files per job: {files_per_job}")

        # Create manifest
        manifest_dir = Path("cache/s3_delete")
        manifest_path = (
            manifest_dir / f"delete_manifest_{relative_path.replace('/', '_')}.json"
        )
        create_delete_manifest(keys, manifest_path)

        # Submit SLURM job
        print("\n3. Submitting SLURM job...")
        job_id = submit_slurm_delete_job(manifest_path, num_jobs, files_per_job)

        if job_id:
            print(f"\n   Submitted job: {job_id}")
            print("\n   Monitor with: squeue -j", job_id)
            print("   Cancel with:  scancel", job_id)
        else:
            print("\n   Failed to submit SLURM job")
            sys.exit(1)
    else:
        print("\n2. Deleting with local threads...")
        success, errors = delete_with_threads(keys, args.workers)

        print("\n" + "=" * 80)
        print("DELETION COMPLETE")
        print("=" * 80)
        print(f"  Successful: {success}")
        print(f"  Errors:     {errors}")

        if errors > 0:
            sys.exit(1)

    print("=" * 80)


if __name__ == "__main__":
    main()
