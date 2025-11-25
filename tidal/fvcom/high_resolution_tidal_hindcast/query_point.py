"""
Query tidal data by geographic point.

This script provides a simple CLI for querying tidal parquet data by lat/lon coordinates.
It auto-discovers the latest manifest from S3 or HPC, finds the nearest data point,
and displays the parquet data.

Usage:
    # Query from S3 (default)
    python query_point.py --lat 60.73 --lon -151.43

    # Query from S3 with specific AWS profile
    python query_point.py --lat 60.73 --lon -151.43 --aws-profile my-profile

    # Query from HPC local filesystem
    python query_point.py --lat 60.73 --lon -151.43 --use-hpc

    # Query specific version
    python query_point.py --lat 60.73 --lon -151.43 --manifest-version 1.0.0

    # Show more rows
    python query_point.py --lat 60.73 --lon -151.43 --head 20
"""

import argparse
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from config import config


def parse_semver(version_str: str) -> Tuple[int, int, int]:
    """Parse semantic version string into tuple for comparison."""
    match = re.match(r"(\d+)\.(\d+)\.(\d+)", version_str)
    if not match:
        raise ValueError(f"Invalid semver: {version_str}")
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def find_latest_manifest_hpc(base_path: str) -> Optional[Path]:
    """
    Find the latest manifest on HPC filesystem using semver traversal.

    Parameters
    ----------
    base_path : str
        HPC base path (e.g., /projects/hindcastra/Tidal/datasets/high_resolution_tidal_hindcast)

    Returns
    -------
    Path or None
        Path to latest manifest file, or None if not found
    """
    manifests_dir = Path(base_path) / "manifests"

    if not manifests_dir.exists():
        print(f"  Manifests directory not found: {manifests_dir}")
        return None

    # Find all version directories (v1.0.0, v1.0.1, etc.)
    version_dirs = []
    for d in manifests_dir.iterdir():
        if d.is_dir() and d.name.startswith("v"):
            try:
                version = parse_semver(d.name[1:])  # Remove 'v' prefix
                version_dirs.append((d, version))
            except ValueError:
                continue

    # Sort by version descending
    version_dirs.sort(key=lambda x: x[1], reverse=True)

    for version_dir, _ in version_dirs:
        # Find all manifest files in this version directory
        manifest_files = []
        for f in version_dir.glob("manifest_*.json"):
            match = re.search(r"manifest_(\d+\.\d+\.\d+)\.json", f.name)
            if match:
                try:
                    version = parse_semver(match.group(1))
                    manifest_files.append((f, version))
                except ValueError:
                    continue

        # Sort by version descending
        manifest_files.sort(key=lambda x: x[1], reverse=True)

        if manifest_files:
            manifest_file = manifest_files[0][0]
            print(f"  Found manifest: {manifest_file}")
            return manifest_file

    return None


def find_latest_manifest_s3(
    bucket: str,
    prefix: str,
    aws_profile: Optional[str] = None,
    staging_path: Optional[str] = None,
) -> Optional[Path]:
    """
    Find the latest manifest on S3 using semver traversal.

    Parameters
    ----------
    bucket : str
        S3 bucket name
    prefix : str
        S3 prefix (e.g., 'us-tidal')
    aws_profile : str, optional
        AWS profile name to use
    staging_path : str, optional
        Local path to stage downloaded files

    Returns
    -------
    Path or None
        Path to downloaded manifest file, or None if not found
    """
    import boto3

    # Create S3 client with optional profile
    if aws_profile:
        session = boto3.Session(profile_name=aws_profile)
        s3 = session.client("s3")
    else:
        s3 = boto3.client("s3")

    manifest_prefix = f"{prefix}/manifest/"

    try:
        # List all objects under manifest prefix
        paginator = s3.get_paginator("list_objects_v2")
        manifest_files = []

        for page in paginator.paginate(Bucket=bucket, Prefix=manifest_prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                # Match manifest files: manifest/{version}/manifest_{version}.json
                match = re.search(
                    r"manifest/(\d+\.\d+\.\d+)/manifest_(\d+\.\d+\.\d+)\.json", key
                )
                if match:
                    manifest_files.append((key, parse_semver(match.group(2))))

        if not manifest_files:
            print(f"  No manifests found in s3://{bucket}/{manifest_prefix}")
            return None

        # Sort by version (descending) and get latest
        manifest_files.sort(key=lambda x: x[1], reverse=True)
        latest_key = manifest_files[0][0]
        latest_version = ".".join(map(str, manifest_files[0][1]))

        print(f"  Found latest manifest: s3://{bucket}/{latest_key}")

        # Download to staging path
        if staging_path:
            local_dir = Path(staging_path)
        else:
            local_dir = Path(tempfile.gettempdir()) / "tidal_manifest_cache"

        local_dir.mkdir(parents=True, exist_ok=True)
        local_file = local_dir / f"manifest_{latest_version}.json"

        # Check if already cached
        if local_file.exists():
            print(f"  Using cached manifest: {local_file}")
        else:
            print(f"  Downloading manifest to: {local_file}")
            s3.download_file(bucket, latest_key, str(local_file))

        return local_file

    except Exception as e:
        print(f"  Error accessing S3: {e}")
        return None


def download_parquet_from_s3(
    bucket: str,
    key: str,
    aws_profile: Optional[str] = None,
    staging_path: Optional[str] = None,
) -> Optional[Path]:
    """
    Download a parquet file from S3.

    Parameters
    ----------
    bucket : str
        S3 bucket name
    key : str
        S3 object key
    aws_profile : str, optional
        AWS profile name
    staging_path : str, optional
        Local staging directory

    Returns
    -------
    Path or None
        Path to downloaded file
    """
    import boto3

    if aws_profile:
        session = boto3.Session(profile_name=aws_profile)
        s3 = session.client("s3")
    else:
        s3 = boto3.client("s3")

    # Create local path
    if staging_path:
        local_dir = Path(staging_path) / "parquet_cache"
    else:
        local_dir = Path(tempfile.gettempdir()) / "tidal_parquet_cache"

    # Use hash of key for local filename to avoid deep directory structures
    key_hash = hashlib.md5(key.encode()).hexdigest()[:12]
    filename = Path(key).name
    local_file = local_dir / f"{key_hash}_{filename}"

    local_dir.mkdir(parents=True, exist_ok=True)

    if local_file.exists():
        print(f"  Using cached parquet: {local_file}")
    else:
        print(f"  Downloading: s3://{bucket}/{key}")
        s3.download_file(bucket, key, str(local_file))
        print(f"  Saved to: {local_file}")

    return local_file


def main():
    # Get defaults from config
    storage_config = config["storage"]
    default_hpc_base = storage_config["hpc_base_path"]
    default_s3_bucket = storage_config["s3_bucket"]
    default_s3_prefix = storage_config["s3_prefix"]

    parser = argparse.ArgumentParser(
        description="Query tidal data by geographic point",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Query from S3 (default)
    python query_point.py --lat 60.73 --lon -151.43

    # Query from HPC local filesystem
    python query_point.py --lat 60.73 --lon -151.43 --use-hpc

    # Query with specific AWS profile
    python query_point.py --lat 60.73 --lon -151.43 --aws-profile nrel-aws

    # Show 20 rows of data
    python query_point.py --lat 60.73 --lon -151.43 --head 20

Test coordinates:
    Cook Inlet:        --lat 60.7320786 --lon -151.4315796
    Piscataqua River:  --lat 43.0521126 --lon -70.7007828
        """,
    )

    # Required arguments
    parser.add_argument(
        "--lat", type=float, required=True, help="Query latitude (decimal degrees)"
    )
    parser.add_argument(
        "--lon", type=float, required=True, help="Query longitude (decimal degrees)"
    )

    # Data source options
    parser.add_argument(
        "--use-hpc",
        action="store_true",
        help="Use HPC local filesystem instead of S3",
    )
    parser.add_argument(
        "--hpc-base-path",
        type=str,
        default=default_hpc_base,
        help=f"HPC base path (default: {default_hpc_base})",
    )

    # AWS options
    parser.add_argument(
        "--aws-profile",
        type=str,
        default=None,
        help="AWS profile name for S3 access",
    )
    parser.add_argument(
        "--staging-path",
        type=str,
        default=None,
        help="Local path for staging downloaded files",
    )

    # S3 configuration
    parser.add_argument(
        "--s3-bucket",
        type=str,
        default=default_s3_bucket,
        help=f"S3 bucket name (default: {default_s3_bucket})",
    )
    parser.add_argument(
        "--s3-prefix",
        type=str,
        default=default_s3_prefix,
        help=f"S3 prefix (default: {default_s3_prefix})",
    )

    # Version options
    parser.add_argument(
        "--manifest-version",
        type=str,
        default=None,
        help="Specific manifest version to use (default: latest)",
    )
    parser.add_argument(
        "--data-version",
        type=str,
        default=None,
        help="Specific data version to use (default: latest for location)",
    )

    # Output options
    parser.add_argument(
        "--head",
        type=int,
        default=10,
        help="Number of rows to display (default: %(default)s)",
    )
    parser.add_argument(
        "--max-distance-km",
        type=float,
        default=None,
        help="Maximum distance in km to accept (default: no limit)",
    )
    parser.add_argument(
        "--info-only",
        action="store_true",
        help="Only show point info, don't load parquet data",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Tidal Data Point Query")
    print("=" * 70)
    print(f"\nQuery coordinates: ({args.lat}, {args.lon})")

    # Find manifest
    print("\nLocating manifest...")
    if args.use_hpc:
        print(f"  Source: HPC filesystem ({args.hpc_base_path})")
        manifest_path = find_latest_manifest_hpc(args.hpc_base_path)
    else:
        print(f"  Source: S3 (s3://{args.s3_bucket}/{args.s3_prefix})")
        manifest_path = find_latest_manifest_s3(
            args.s3_bucket,
            args.s3_prefix,
            aws_profile=args.aws_profile,
            staging_path=args.staging_path,
        )

    if manifest_path is None:
        print("\nERROR: Could not find manifest")
        return 1

    # Load manifest and create query interface
    print("\nLoading manifest...")
    from query_tidal_manifest import TidalManifestQuery

    query = TidalManifestQuery(manifest_path)

    # Query for nearest point
    print("\nSearching for nearest point...")
    result = query.query_nearest_point(
        lat=args.lat,
        lon=args.lon,
        load_details=False,
    )

    if result is None:
        print("\nNo data points found near the query location.")
        return 1

    # Check distance threshold
    if args.max_distance_km and result["distance_km"] > args.max_distance_km:
        print(
            f"\nNearest point is {result['distance_km']:.2f} km away, "
            f"exceeds max distance of {args.max_distance_km} km"
        )
        return 1

    # Display result
    print("\n" + "-" * 70)
    print("NEAREST POINT FOUND")
    print("-" * 70)
    print(f"  Face ID:      {result['point']['face_id']}")
    print(f"  Latitude:     {result['point']['lat']}")
    print(f"  Longitude:    {result['point']['lon']}")
    print(f"  Distance:     {result['distance_km']:.4f} km from query point")
    print(f"  Location:     {result['location']}")
    print(f"  Grid ID:      {result['grid_id']}")

    # Get version info
    version_info = query.get_location_version_info(result["location"])
    data_version = args.data_version or version_info["latest_version"]
    print(f"  Data Version: {data_version}")

    # Get file path
    relative_path = result["point"]["file_path"]
    print(f"\n  Relative path: {relative_path}")

    if args.use_hpc:
        full_path = query.get_hpc_path(relative_path)
        print(f"  HPC path:      {full_path}")
    else:
        s3_uri = query.get_s3_uri(relative_path)
        print(f"  S3 URI:        {s3_uri}")

    if args.info_only:
        print("\n(--info-only specified, skipping parquet data load)")
        return 0

    # Load and display parquet data
    print("\n" + "-" * 70)
    print("PARQUET DATA")
    print("-" * 70)

    try:
        if args.use_hpc:
            parquet_path = Path(full_path)
            if not parquet_path.exists():
                print(f"\nERROR: File not found: {parquet_path}")
                return 1
        else:
            # Download from S3
            parquet_path = download_parquet_from_s3(
                args.s3_bucket,
                f"{args.s3_prefix}/{relative_path}",
                aws_profile=args.aws_profile,
                staging_path=args.staging_path,
            )

        print(f"\nLoading: {parquet_path}")
        df = pd.read_parquet(parquet_path)

        print(f"\nShape: {df.shape[0]} rows x {df.shape[1]} columns")
        print(f"Columns: {list(df.columns)}")
        print(f"\nFirst {args.head} rows:")
        print(df.head(args.head).to_string())

        # Show basic stats
        print("\nData summary:")
        numeric_cols = df.select_dtypes(include=["number"]).columns
        if len(numeric_cols) > 0:
            print(df[numeric_cols].describe().to_string())

    except Exception as e:
        print(f"\nERROR loading parquet: {e}")
        return 1

    print("\n" + "=" * 70)
    print("Query complete!")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    exit(main())
