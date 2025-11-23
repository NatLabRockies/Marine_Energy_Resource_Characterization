#!/usr/bin/env python3
"""
Upload files from SQLite manifest by batch index.

This script is called by SLURM array jobs. Each job processes a batch of files
based on SLURM_ARRAY_TASK_ID and files_per_job configuration.
"""

import argparse
import os
import sqlite3
import sys
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


def get_files_for_batch(manifest_path, batch_index, files_per_job):
    """
    Get files to process for this batch.

    Args:
        manifest_path: Path to SQLite manifest
        batch_index: Batch index (from SLURM_ARRAY_TASK_ID)
        files_per_job: Number of files per batch

    Returns:
        List of file records (tuples)
    """
    conn = sqlite3.connect(manifest_path)
    cursor = conn.cursor()

    # Calculate range
    start_index = batch_index * files_per_job
    end_index = start_index + files_per_job

    # Get files in this batch that are pending
    cursor.execute(
        """
        SELECT file_index, local_path, s3_destination, file_size_bytes
        FROM files
        WHERE file_index >= ? AND file_index < ?
        AND upload_status = 'pending'
        ORDER BY file_index
        """,
        (start_index, end_index),
    )

    files = cursor.fetchall()
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
    conn = sqlite3.connect(manifest_path, timeout=30.0)
    cursor = conn.cursor()

    timestamp = datetime.now().isoformat()

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

    for file_index, local_path, s3_destination, file_size in files:
        print(f"\n{'=' * 80}")
        print(f"File {file_index + 1} / {len(files)} in batch")

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
