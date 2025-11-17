#!/usr/bin/env python3
"""
Single file S3 upload script with checksum verification.

This script uploads a single file to S3 with optional features:
- Check if file exists in S3 (skip if exists)
- Checksum verification after upload
- Dry-run mode for testing

Usage:
    python upload_to_s3.py <local_file> <s3_destination> [options]
"""

import argparse
import hashlib
import os
import sys
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

# S3 Configuration - Global constants
S3_BUCKET = "oedi-data-drop"
S3_PROFILE = "us-tidal"
S3_BASE_PATH = "us-tidal"


def calculate_file_size(file_path):
    """Calculate file size in human-readable format."""
    size_bytes = os.path.getsize(file_path)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def calculate_md5(file_path, chunk_size=8192 * 1024):
    """
    Calculate MD5 hash of a file efficiently for large files.

    Args:
        file_path: Path to the file
        chunk_size: Size of chunks to read (default 8MB)

    Returns:
        MD5 hex digest string
    """
    md5_hash = hashlib.md5()
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()


def s3_file_exists(s3_client, bucket, key):
    """
    Check if a file exists in S3.

    Args:
        s3_client: Boto3 S3 client
        bucket: S3 bucket name
        key: S3 object key

    Returns:
        True if file exists, False otherwise
    """
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] == "404":
            return False
        else:
            raise


def verify_upload(s3_client, bucket, key, local_md5):
    """
    Verify uploaded file matches local file using MD5 checksum.

    Args:
        s3_client: Boto3 S3 client
        bucket: S3 bucket name
        key: S3 object key
        local_md5: Local file MD5 hash

    Returns:
        True if checksums match, False otherwise
    """
    try:
        response = s3_client.head_object(Bucket=bucket, Key=key)
        # S3 ETag is MD5 for single-part uploads (files < 5GB typically)
        s3_etag = response["ETag"].strip('"')

        if s3_etag == local_md5:
            return True
        else:
            print("Warning: Checksum mismatch!", file=sys.stderr)
            print(f"  Local MD5:  {local_md5}", file=sys.stderr)
            print(f"  S3 ETag:    {s3_etag}", file=sys.stderr)
            return False
    except ClientError as e:
        print(f"Error verifying upload: {e}", file=sys.stderr)
        return False


def upload_file(
    local_file,
    s3_destination,
    skip_if_exists=True,
    verify_checksum=True,
    dry_run=False,
):
    """
    Upload a file to S3 with optional features.

    Args:
        local_file: Path to local file
        s3_destination: S3 destination path (without bucket name)
        skip_if_exists: Skip upload if file exists in S3
        verify_checksum: Verify upload with MD5 checksum
        dry_run: Print actions without executing

    Returns:
        0 on success, 1 on failure
    """
    # Validate local file exists
    if not os.path.exists(local_file):
        print(f"Error: Local file does not exist: {local_file}", file=sys.stderr)
        return 1

    # Construct full S3 key with base path
    s3_key = f"{S3_BASE_PATH}/{s3_destination}"

    # Print upload details
    file_size = calculate_file_size(local_file)
    print(f"Local file:  {local_file}")
    print(f"File size:   {file_size}")
    print(f"S3 bucket:   {S3_BUCKET}")
    print(f"S3 key:      {s3_key}")
    print(f"S3 profile:  {S3_PROFILE}")
    print(f"Full S3 URI: s3://{S3_BUCKET}/{s3_key}")

    if dry_run:
        print("\n[DRY RUN] Would upload file to S3")
        return 0

    # Initialize S3 client
    try:
        session = boto3.Session(profile_name=S3_PROFILE)
        s3_client = session.client("s3")
    except Exception as e:
        print(f"Error initializing S3 client: {e}", file=sys.stderr)
        return 1

    # Check if file exists in S3
    if skip_if_exists:
        print("\nChecking if file exists in S3...")
        if s3_file_exists(s3_client, S3_BUCKET, s3_key):
            print(f"File already exists in S3, skipping upload: {s3_key}")
            return 0
        else:
            print("File does not exist in S3, proceeding with upload")

    # Calculate MD5 if verification is requested
    local_md5 = None
    if verify_checksum:
        print("\nCalculating local file MD5...")
        local_md5 = calculate_md5(local_file)
        print(f"Local MD5: {local_md5}")

    # Upload file
    print("\nUploading file to S3...")
    try:
        s3_client.upload_file(local_file, S3_BUCKET, s3_key)
        print("Upload successful!")
    except Exception as e:
        print(f"Error uploading file: {e}", file=sys.stderr)
        return 1

    # Verify upload
    if verify_checksum and local_md5:
        print("\nVerifying upload with checksum...")
        if verify_upload(s3_client, S3_BUCKET, s3_key, local_md5):
            print("Checksum verification successful!")
        else:
            print("Checksum verification failed!", file=sys.stderr)
            return 1

    print(f"\nSuccessfully uploaded: {local_file}")
    print(f"S3 URI: s3://{S3_BUCKET}/{s3_key}")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Upload a single file to S3 with optional verification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic upload
  python upload_to_s3.py /path/to/file.nc AK_cook_inlet/0.4.0/b1_vap/2005/file.nc

  # Dry run
  python upload_to_s3.py /path/to/file.nc AK_cook_inlet/0.4.0/b1_vap/2005/file.nc --dry-run

  # Force upload (skip exists check)
  python upload_to_s3.py /path/to/file.nc AK_cook_inlet/0.4.0/b1_vap/2005/file.nc --no-skip-if-exists

  # Skip checksum verification
  python upload_to_s3.py /path/to/file.nc AK_cook_inlet/0.4.0/b1_vap/2005/file.nc --no-verify-checksum
        """,
    )

    parser.add_argument(
        "local_file",
        help="Path to local file to upload",
    )

    parser.add_argument(
        "s3_destination",
        help="S3 destination path (e.g., AK_cook_inlet/0.4.0/b1_vap/2005/file.nc)",
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
        help="Print actions without executing upload",
    )

    args = parser.parse_args()

    # Execute upload
    exit_code = upload_file(
        local_file=args.local_file,
        s3_destination=args.s3_destination,
        skip_if_exists=args.skip_if_exists,
        verify_checksum=args.verify_checksum,
        dry_run=args.dry_run,
    )

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
