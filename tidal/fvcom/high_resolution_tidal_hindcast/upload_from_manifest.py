#!/usr/bin/env python3
"""
Upload files from SQLite manifest by batch index.

This script is called by SLURM array jobs. Each job processes a batch of files
based on SLURM_ARRAY_TASK_ID and files_per_job configuration.
"""

import argparse
import os
import random
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.exceptions import ClientError

# Import from upload_to_s3
from upload_to_s3 import (
    S3_BUCKET,
    S3_BASE_PATH,
    S3_PROFILE,
    TRANSFER_CONFIG,
    calculate_s3_etag,
)

RANDOM_START_DELAY_MAX = 60.0

# SQLite retry configuration for network filesystems
SQLITE_TIMEOUT = 60.0  # seconds
SQLITE_MAX_RETRIES = 5
SQLITE_RETRY_BASE_DELAY = 1.0  # seconds


def connect_with_retry(manifest_path, max_retries=SQLITE_MAX_RETRIES):
    """
    Connect to SQLite database with retry logic for network filesystems.

    On HPC systems with shared filesystems (NFS, Lustre), SQLite can encounter
    locking protocol errors when many jobs access the database simultaneously.
    This function implements exponential backoff with jitter to handle these
    transient failures.

    Args:
        manifest_path: Path to SQLite manifest database
        max_retries: Maximum number of connection attempts

    Returns:
        sqlite3.Connection object

    Raises:
        sqlite3.OperationalError: If all retries are exhausted
    """
    last_error = None

    for attempt in range(max_retries):
        try:
            # Use longer timeout and enable WAL mode for better concurrency
            conn = sqlite3.connect(
                manifest_path,
                timeout=SQLITE_TIMEOUT,
                isolation_level="DEFERRED",
            )
            # Enable WAL mode for better concurrent read performance
            # This is safe to call multiple times - SQLite handles it gracefully
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=60000")  # 60 second busy timeout
            return conn

        except sqlite3.OperationalError as e:
            last_error = e
            if attempt < max_retries - 1:
                # Exponential backoff with jitter
                delay = SQLITE_RETRY_BASE_DELAY * (2**attempt) + random.uniform(0, 1)
                print(
                    f"  SQLite connection attempt {attempt + 1}/{max_retries} failed: {e}",
                    file=sys.stderr,
                )
                print(f"  Retrying in {delay:.2f}s...", file=sys.stderr)
                time.sleep(delay)

    # All retries exhausted
    raise sqlite3.OperationalError(
        f"Failed to connect after {max_retries} attempts: {last_error}"
    )


def get_files_for_batch(manifest_path, batch_index, files_per_job):
    """
    Get files to process for this batch.

    Uses OFFSET/LIMIT on pending files to correctly handle sparse indices
    when some files have been skipped.

    Args:
        manifest_path: Path to SQLite manifest
        batch_index: Batch index (from SLURM_ARRAY_TASK_ID)
        files_per_job: Number of files per batch

    Returns:
        List of file records (tuples)
    """
    conn = connect_with_retry(manifest_path)
    cursor = conn.cursor()

    # Calculate offset for this batch among pending files only
    offset = batch_index * files_per_job

    try:
        # Get pending files using OFFSET/LIMIT to handle sparse indices
        # This correctly handles cases where some files are 'skipped'
        cursor.execute(
            """
            SELECT file_index, local_path, s3_destination, file_size_bytes
            FROM files
            WHERE upload_status = 'pending'
            ORDER BY file_index
            LIMIT ? OFFSET ?
            """,
            (files_per_job, offset),
        )

        files = cursor.fetchall()
    finally:
        conn.close()

    return files


def update_file_status(manifest_path, file_index, status, etag=None):
    """
    Update upload status in manifest.

    Args:
        manifest_path: Path to SQLite manifest
        file_index: File index
        status: Upload status ('completed', 'failed')
        etag: S3 ETag (optional)
    """
    conn = connect_with_retry(manifest_path)
    cursor = conn.cursor()

    timestamp = datetime.now().isoformat()

    try:
        cursor.execute(
            """
            UPDATE files
            SET upload_status = ?,
                etag = ?,
                upload_timestamp = ?
            WHERE file_index = ?
            """,
            (status, etag, timestamp, file_index),
        )

        conn.commit()
    finally:
        conn.close()


def upload_single_file(s3_client, local_path, s3_destination, verify_checksum=True):
    """
    Upload a single file to S3.

    Args:
        s3_client: Boto3 S3 client
        local_path: Local file path
        s3_destination: S3 destination (without bucket)
        verify_checksum: Whether to verify with checksum

    Returns:
        (success: bool, etag: str or None)
    """
    # Construct full S3 key
    s3_key = f"{S3_BASE_PATH}/{s3_destination}"

    print(f"\nUploading: {Path(local_path).name}")
    print(f"  Local:  {local_path}")
    print(f"  S3 key: {s3_key}")

    # Check if file exists
    if not os.path.exists(local_path):
        print("  ERROR: File not found!", file=sys.stderr)
        return False, None

    # Calculate expected ETag if verification requested
    expected_etag = None
    if verify_checksum:
        print("  Calculating ETag...")
        expected_etag = calculate_s3_etag(
            local_path, TRANSFER_CONFIG.multipart_chunksize
        )

    # Upload file
    try:
        file_size = os.path.getsize(local_path)
        print(f"  Size: {file_size / (1024**2):.2f} MB")

        if file_size > TRANSFER_CONFIG.multipart_threshold:
            num_parts = file_size // TRANSFER_CONFIG.multipart_chunksize + 1
            print(f"  Multipart: {num_parts} parts")

        s3_client.upload_file(local_path, S3_BUCKET, s3_key, Config=TRANSFER_CONFIG)
        print("  Upload: SUCCESS")

    except Exception as e:
        print(f"  Upload: FAILED - {e}", file=sys.stderr)
        return False, None

    # Verify upload
    if verify_checksum and expected_etag:
        try:
            response = s3_client.head_object(Bucket=S3_BUCKET, Key=s3_key)
            s3_etag = response["ETag"].strip('"')

            if s3_etag == expected_etag:
                print("  Verify: SUCCESS")
                return True, s3_etag
            else:
                print("  Verify: FAILED - ETag mismatch", file=sys.stderr)
                print(f"    Expected: {expected_etag}", file=sys.stderr)
                print(f"    Got:      {s3_etag}", file=sys.stderr)
                return False, None

        except ClientError as e:
            print(f"  Verify: FAILED - {e}", file=sys.stderr)
            return False, None
    else:
        # No verification, consider success
        return True, None


def main():
    parser = argparse.ArgumentParser(
        description="Upload files from manifest by batch index"
    )

    parser.add_argument("manifest_path", help="Path to SQLite manifest file")

    parser.add_argument(
        "batch_index", type=int, help="Batch index (from SLURM_ARRAY_TASK_ID)"
    )

    parser.add_argument("files_per_job", type=int, help="Number of files per batch")

    parser.add_argument(
        "--no-verify-checksum",
        dest="verify_checksum",
        action="store_false",
        default=True,
        help="Skip checksum verification",
    )

    args = parser.parse_args()

    # Add random startup delay to stagger database access across SLURM jobs
    # This helps prevent "thundering herd" when many jobs start simultaneously
    startup_delay = random.uniform(0, RANDOM_START_DELAY_MAX)  # 0-5 second random delay
    print(f"Startup delay: {startup_delay:.2f}s (staggering database access)")
    time.sleep(startup_delay)

    # Get files for this batch
    print(f"Batch {args.batch_index}: Processing up to {args.files_per_job} files")
    files = get_files_for_batch(
        args.manifest_path, args.batch_index, args.files_per_job
    )

    if not files:
        print("No pending files in this batch")
        return 0

    print(f"Found {len(files)} pending files to upload")

    # Initialize S3 client
    try:
        session = boto3.Session(profile_name=S3_PROFILE)
        s3_client = session.client("s3")
    except Exception as e:
        print(f"Error initializing S3 client: {e}", file=sys.stderr)
        return 1

    # Upload each file (no real-time DB updates to avoid lock contention)
    success_count = 0
    failure_count = 0
    failed_files = []  # Track failures for logging
    total_files = len(files)

    for batch_idx, (file_index, local_path, s3_destination, file_size) in enumerate(
        files
    ):
        print(f"\n{'=' * 80}")
        print(
            f"File {batch_idx + 1} / {total_files} in batch (global index: {file_index})"
        )

        success, etag = upload_single_file(
            s3_client, local_path, s3_destination, args.verify_checksum
        )

        if success:
            success_count += 1
        else:
            failure_count += 1
            failed_files.append((file_index, local_path, s3_destination))
            # Log failure to stderr for SLURM to capture
            print(
                f"FAILED: index={file_index} path={local_path} dest={s3_destination}",
                file=sys.stderr,
            )

    # Summary
    print(f"\n{'=' * 80}")
    print(f"Batch {args.batch_index} Summary:")
    print(f"  Successful: {success_count}")
    print(f"  Failed:     {failure_count}")
    print(f"  Total:      {len(files)}")

    if failed_files:
        print("\nFailed files logged to stderr for SLURM capture")

    # Return non-zero if any failures (SLURM will track this)
    return 1 if failure_count > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
