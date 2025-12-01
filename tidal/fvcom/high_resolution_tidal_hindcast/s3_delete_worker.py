#!/usr/bin/env python3
"""
S3 Delete Worker - Process a chunk of files from a delete manifest.

This script is called by s3_delete_array.sbatch to delete a subset of files
from a larger manifest.

Includes rate limiting protection:
- Random startup jitter to stagger concurrent jobs
- Exponential backoff with retry on SlowDown errors
- Delays between batches to avoid overwhelming S3

Usage:
    python s3_delete_worker.py <manifest_path> <task_id> <files_per_job> [options]

Example:
    python s3_delete_worker.py cache/s3_delete/manifest.json 0 1000 \\
        --bucket oedi-data-drop --profile us-tidal
"""

import argparse
import json
import random
import sys
import time

import boto3
from botocore.exceptions import ClientError

# Rate limiting configuration
MAX_RETRIES = 8  # Max retry attempts per batch
BASE_DELAY = 1.0  # Base delay in seconds for exponential backoff
MAX_DELAY = 120.0  # Maximum delay between retries (2 minutes)
BATCH_DELAY = 0.5  # Delay between successful batches (seconds)
STARTUP_JITTER_MAX = 30  # Max random startup delay (seconds)


def load_manifest(manifest_path: str) -> dict:
    """Load the delete manifest from JSON file."""
    with open(manifest_path) as f:
        return json.load(f)


def get_keys_for_task(
    all_keys: list[str], task_id: int, files_per_job: int
) -> list[str]:
    """
    Get the subset of keys for this SLURM array task.

    Args:
        all_keys: Full list of S3 keys
        task_id: SLURM array task ID (0-indexed)
        files_per_job: Number of files per task

    Returns:
        List of keys for this task
    """
    start_idx = task_id * files_per_job
    end_idx = min(start_idx + files_per_job, len(all_keys))

    return all_keys[start_idx:end_idx]


def is_retryable_error(error: ClientError) -> bool:
    """Check if an S3 error is retryable (rate limiting or transient)."""
    error_code = error.response.get("Error", {}).get("Code", "")
    return error_code in ("SlowDown", "ServiceUnavailable", "InternalError")


def delete_single_batch_with_retry(
    s3_client, bucket: str, batch: list[str]
) -> tuple[int, int, list[str]]:
    """
    Delete a single batch (up to 1000 keys) with exponential backoff retry.

    Args:
        s3_client: Boto3 S3 client
        bucket: S3 bucket name
        batch: List of S3 object keys to delete (max 1000)

    Returns:
        Tuple of (success_count, error_count, error_messages)
    """
    delete_request = {"Objects": [{"Key": k} for k in batch], "Quiet": True}

    for attempt in range(MAX_RETRIES):
        try:
            response = s3_client.delete_objects(Bucket=bucket, Delete=delete_request)

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
                    f"    Rate limited, retrying in {delay:.1f}s (attempt {attempt + 1}/{MAX_RETRIES})"
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


def delete_objects_batch(
    s3_client, bucket: str, keys: list[str]
) -> tuple[int, int, list[str]]:
    """
    Delete a batch of S3 objects using delete_objects API with retry logic.

    Args:
        s3_client: Boto3 S3 client
        bucket: S3 bucket name
        keys: List of S3 object keys to delete

    Returns:
        Tuple of (success_count, error_count, error_messages)
    """
    success_count = 0
    error_count = 0
    errors = []

    # S3 delete_objects can handle up to 1000 objects per call
    num_batches = (len(keys) + 999) // 1000
    for batch_num, i in enumerate(range(0, len(keys), 1000)):
        batch = keys[i : i + 1000]

        batch_success, batch_errors, batch_msgs = delete_single_batch_with_retry(
            s3_client, bucket, batch
        )

        success_count += batch_success
        error_count += batch_errors
        errors.extend(batch_msgs)

        # Progress indicator
        if (batch_num + 1) % 5 == 0 or batch_num == num_batches - 1:
            print(
                f"  Batch {batch_num + 1}/{num_batches}: {success_count} deleted, {error_count} errors"
            )

        # Small delay between batches to avoid overwhelming S3
        if batch_num < num_batches - 1:
            time.sleep(BATCH_DELAY)

    return success_count, error_count, errors


def main():
    parser = argparse.ArgumentParser(
        description="S3 delete worker for SLURM array jobs"
    )

    parser.add_argument("manifest", help="Path to JSON manifest file")
    parser.add_argument("task_id", type=int, help="SLURM array task ID")
    parser.add_argument("files_per_job", type=int, help="Number of files per job")
    parser.add_argument("--bucket", required=True, help="S3 bucket name")
    parser.add_argument("--profile", required=True, help="AWS profile name")

    args = parser.parse_args()

    # Random startup delay to stagger concurrent SLURM jobs
    # This helps prevent all 500 jobs from hitting S3 simultaneously
    startup_delay = random.uniform(0, STARTUP_JITTER_MAX)
    print(f"Startup jitter: waiting {startup_delay:.1f}s to stagger requests...")
    time.sleep(startup_delay)

    print(f"Loading manifest: {args.manifest}")
    try:
        manifest = load_manifest(args.manifest)
    except Exception as e:
        print(f"Error loading manifest: {e}", file=sys.stderr)
        sys.exit(1)

    all_keys = manifest.get("keys", [])
    print(f"Total keys in manifest: {len(all_keys)}")

    # Get keys for this task
    keys = get_keys_for_task(all_keys, args.task_id, args.files_per_job)

    if not keys:
        print(f"No keys for task {args.task_id} (may be out of range)")
        sys.exit(0)

    print(f"Task {args.task_id}: Processing {len(keys)} keys")
    print(f"  First key: {keys[0]}")
    print(f"  Last key:  {keys[-1]}")

    # Create S3 client
    try:
        session = boto3.Session(profile_name=args.profile)
        s3_client = session.client("s3")
    except Exception as e:
        print(f"Error creating S3 client: {e}", file=sys.stderr)
        sys.exit(1)

    # Delete objects
    print(f"\nDeleting {len(keys)} objects from s3://{args.bucket}/...")
    success, errors, error_msgs = delete_objects_batch(s3_client, args.bucket, keys)

    # Report results
    print("\nResults:")
    print(f"  Successful: {success}")
    print(f"  Errors:     {errors}")

    if error_msgs:
        print("\nErrors encountered:")
        for msg in error_msgs[:10]:
            print(f"  {msg}")
        if len(error_msgs) > 10:
            print(f"  ... and {len(error_msgs) - 10} more errors")

    if errors > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
