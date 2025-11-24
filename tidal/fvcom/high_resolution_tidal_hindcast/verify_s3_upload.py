#!/usr/bin/env python3
"""
Verify S3 upload status and file integrity.

This script checks:
1. If the file exists in S3
2. The S3 file's ETag, size, and metadata
3. Compares with local file if accessible
4. Calculates correct multipart ETag for comparison
"""

import argparse
import hashlib
import os
import sys
from pathlib import Path

import boto3
from botocore.exceptions import ClientError


def calculate_multipart_etag(file_path: Path, chunk_size_mb: int = 500) -> tuple[str, int]:
    """
    Calculate the ETag for a multipart upload.

    AWS uses MD5 hashes of each part, then MD5 hash of concatenated hashes.

    Args:
        file_path: Path to local file
        chunk_size_mb: Chunk size in MB (default 500MB, matching upload script)

    Returns:
        Tuple of (etag_string, num_parts)
    """
    chunk_size = chunk_size_mb * 1024 * 1024
    md5_hashes = []

    print(f"Calculating multipart ETag (chunk size: {chunk_size_mb}MB)...")

    with open(file_path, 'rb') as f:
        part_num = 0
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            part_num += 1
            md5_hashes.append(hashlib.md5(chunk).digest())

            if part_num % 100 == 0:
                print(f"  Processed {part_num} parts...")

    # Combine all part hashes
    combined_hash = hashlib.md5(b''.join(md5_hashes)).hexdigest()
    etag = f"{combined_hash}-{len(md5_hashes)}"

    print(f"  Total parts: {len(md5_hashes)}")
    print(f"  Calculated ETag: {etag}")

    return etag, len(md5_hashes)


def check_s3_file(bucket: str, key: str, local_path: Path = None, profile: str = None):
    """
    Check if file exists in S3 and get its metadata.

    Args:
        bucket: S3 bucket name
        key: S3 object key
        local_path: Optional local file path for comparison
        profile: AWS profile name to use
    """
    session = boto3.Session(profile_name=profile) if profile else boto3.Session()
    s3_client = session.client('s3')

    print("\n" + "="*80)
    print("S3 File Verification")
    print("="*80)
    print(f"Bucket: {bucket}")
    print(f"Key:    {key}")
    print()

    try:
        # Get object metadata
        response = s3_client.head_object(Bucket=bucket, Key=key)

        print("✓ File exists in S3")
        print(f"\nS3 Metadata:")
        print(f"  Size:          {response['ContentLength']:,} bytes ({response['ContentLength'] / (1024**3):.2f} GB)")
        etag_value = response['ETag'].strip('"')
        print(f"  ETag:          {etag_value}")
        print(f"  Last Modified: {response['LastModified']}")
        print(f"  Content-Type:  {response.get('ContentType', 'N/A')}")

        if 'Metadata' in response and response['Metadata']:
            print(f"  Custom Metadata:")
            for k, v in response['Metadata'].items():
                print(f"    {k}: {v}")

        s3_etag = response['ETag'].strip('"')
        s3_size = response['ContentLength']

        # Check if it's a multipart upload
        if '-' in s3_etag:
            parts = int(s3_etag.split('-')[1])
            print(f"\n  Multipart Upload: {parts} parts")

        # Compare with local file if provided
        if local_path and local_path.exists():
            print(f"\n" + "-"*80)
            print("Local File Comparison")
            print("-"*80)

            local_size = local_path.stat().st_size
            print(f"Local Size:  {local_size:,} bytes ({local_size / (1024**3):.2f} GB)")
            print(f"S3 Size:     {s3_size:,} bytes ({s3_size / (1024**3):.2f} GB)")

            if local_size == s3_size:
                print("✓ Sizes match")
            else:
                print("✗ Size mismatch!")
                print(f"  Difference: {abs(local_size - s3_size):,} bytes")
                return False

            # Calculate local ETag
            print()
            local_etag, num_parts = calculate_multipart_etag(local_path, chunk_size_mb=500)

            print(f"\nETag Comparison:")
            print(f"  Local:  {local_etag}")
            print(f"  S3:     {s3_etag}")

            if local_etag == s3_etag:
                print("✓ ETags match - upload is valid!")
                return True
            else:
                print("✗ ETag mismatch - file may be corrupted or incomplete")

                # Try different chunk sizes
                print("\nTrying alternative chunk sizes...")
                for chunk_mb in [100, 250, 1000]:
                    alt_etag, alt_parts = calculate_multipart_etag(local_path, chunk_size_mb=chunk_mb)
                    print(f"  {chunk_mb}MB chunks ({alt_parts} parts): {alt_etag}")
                    if alt_etag == s3_etag:
                        print(f"  ✓ Match found with {chunk_mb}MB chunks!")
                        return True

                return False
        else:
            if local_path:
                print(f"\n✗ Local file not found: {local_path}")
            print("\n⚠ Cannot verify integrity without local file")
            return None

    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == '404':
            print("✗ File does not exist in S3")
        else:
            print(f"✗ Error accessing S3: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Verify S3 upload status and file integrity",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check if file exists in S3
  python verify_s3_upload.py --bucket nrel-pds-wtpo --key us-tidal/WA_puget_sound/v1.0.0/hsds/WA_puget_sound.wpto_high_res_tidal.hsds.v1.0.0.h5

  # Verify with local file comparison
  python verify_s3_upload.py --bucket nrel-pds-wtpo --key us-tidal/WA_puget_sound/v1.0.0/hsds/WA_puget_sound.wpto_high_res_tidal.hsds.v1.0.0.h5 \\
      --local-path /kfs2/projects/hindcastra/Tidal/datasets/high_resolution_tidal_hindcast/WA_puget_sound/v1.0.0/hsds/WA_puget_sound.wpto_high_res_tidal.hsds.v1.0.0.h5
        """
    )

    parser.add_argument('--bucket', required=True, help='S3 bucket name')
    parser.add_argument('--key', required=True, help='S3 object key')
    parser.add_argument('--local-path', type=Path, help='Local file path for comparison')
    parser.add_argument('--profile', default='us-tidal', help='AWS profile name (default: us-tidal)')

    args = parser.parse_args()

    result = check_s3_file(args.bucket, args.key, args.local_path, args.profile)

    if result is True:
        print("\n" + "="*80)
        print("✓ VERIFICATION PASSED")
        print("="*80)
        sys.exit(0)
    elif result is False:
        print("\n" + "="*80)
        print("✗ VERIFICATION FAILED")
        print("="*80)
        sys.exit(1)
    else:
        print("\n" + "="*80)
        print("⚠ VERIFICATION INCOMPLETE")
        print("="*80)
        sys.exit(2)


if __name__ == '__main__':
    main()