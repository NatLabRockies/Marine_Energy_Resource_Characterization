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
from src.file_manager import (
    get_output_dirs,
    get_manifest_output_dir,
    validate_manifest_version,
)

# Data level configurations with SLURM time limits (in hours) and file discovery rules
DATA_LEVEL_CONFIG = {
    "00_raw": {
        "time_hours": 6,
        "description": "Raw data",
        "valid_extensions": [".nc"],
        "should_recurse": True,
    },
    "a1_std": {
        "time_hours": 6,
        "description": "Standardized data",
        "valid_extensions": [".nc"],
        "should_recurse": False,
    },
    "a2_std_partition": {
        "time_hours": 6,
        "description": "Standardized partition",
        "valid_extensions": [".nc"],
        "should_recurse": False,
    },
    "b1_vap": {
        "time_hours": 6,
        "description": "Value-added products",
        "valid_extensions": [".nc"],
        "should_recurse": False,
    },
    "b1_vap_daily_compressed": {
        "time_hours": 6,
        "description": "Compressed VAP (daily)",
        "valid_extensions": [".nc"],
        "should_recurse": False,
    },
    "b2_monthly_mean_vap": {
        "time_hours": 6,
        "description": "Monthly mean VAP",
        "valid_extensions": [".nc"],
        "should_recurse": False,
    },
    "b3_yearly_mean_vap": {
        "time_hours": 6,
        "description": "Yearly mean VAP",
        "valid_extensions": [".nc"],
        "should_recurse": False,
    },
    "b1_vap_by_point_partition": {
        "time_hours": 6,
        "description": "VAP partition",
        "valid_extensions": [".parquet"],
        "should_recurse": True,
    },
    "b4_vap_summary_parquet": {
        "time_hours": 6,
        "description": "VAP summary parquet",
        "valid_extensions": None,
        "should_recurse": True,
    },
    "b5_vap_atlas_summary_parquet": {
        "time_hours": 6,
        "description": "VAP atlas summary parquet",
        "valid_extensions": None,
        "should_recurse": True,
    },
    "hsds": {
        "time_hours": 24,
        "description": "HSDS format",
        "valid_extensions": [".h5"],
        "should_recurse": False,
    },
    "manifest": {
        "time_hours": 1,
        "description": "Global manifest (JSON)",
        "valid_extensions": [".json"],
        "should_recurse": True,
        "is_global": True,  # Not location-specific
    },
}

# Location key to output_name mapping from config
LOCATION_MAP = {
    key: spec["output_name"] for key, spec in config["location_specification"].items()
}

# AWS profile for S3 uploads
AWS_PROFILE = "us-tidal"


def get_file_size_human(size_bytes):
    """Convert bytes to human-readable format."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def discover_manifest_files():
    """
    Discover manifest JSON files for upload.

    Validates that config manifest version matches filesystem latest version,
    then discovers all JSON files recursively.

    Returns:
        List of (local_path, relative_path) tuples

    Raises:
        ValueError: If manifest version mismatch between config and filesystem
    """
    # Validate manifest version consistency
    try:
        _, config_version, fs_version, manifest_dir = validate_manifest_version(config)
        print(
            f"Manifest version validated: {config_version} (config) = {fs_version} (filesystem)"
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Get file discovery configuration
    file_config = DATA_LEVEL_CONFIG["manifest"]
    valid_extensions = file_config["valid_extensions"]

    # Discover files recursively
    files = []
    for file_path in manifest_dir.rglob("*"):
        if file_path.is_file():
            if valid_extensions is None or file_path.suffix in valid_extensions:
                relative_path = file_path.relative_to(manifest_dir)
                files.append((str(file_path), str(relative_path)))

    return sorted(files)


def discover_files(location_key, data_levels):
    """
    Discover files for given location and data levels using versioned directories.

    Args:
        location_key: Location key (e.g., 'cook_inlet') - ignored for global data levels like 'manifest'
        data_levels: List of data level directory names (e.g., ['b1_vap', 'hsds'])

    Returns:
        Dictionary mapping data_level to list of (local_path, relative_path) tuples
    """
    # Separate global data levels from location-specific ones
    global_levels = [
        level
        for level in data_levels
        if DATA_LEVEL_CONFIG.get(level, {}).get("is_global", False)
    ]
    location_levels = [level for level in data_levels if level not in global_levels]

    # Validate location only if there are location-specific levels
    if location_levels:
        if location_key not in LOCATION_MAP:
            print(f"Error: Unknown location '{location_key}'", file=sys.stderr)
            print(
                f"Available locations: {', '.join(LOCATION_MAP.keys())}",
                file=sys.stderr,
            )
            sys.exit(1)

    # Validate all data levels
    for level in data_levels:
        if level not in DATA_LEVEL_CONFIG:
            print(f"Error: Unknown data level '{level}'", file=sys.stderr)
            print(
                f"Available data levels: {', '.join(DATA_LEVEL_CONFIG.keys())}",
                file=sys.stderr,
            )
            sys.exit(1)

    files_by_level = {}

    # Handle global data levels (like manifest)
    for data_level in global_levels:
        if data_level == "manifest":
            files = discover_manifest_files()
            files_by_level[data_level] = files
            print(f"Found {len(files)} files in {data_level}/")

    # Handle location-specific data levels
    if location_levels:
        # Get location specification
        location_spec = config["location_specification"][location_key]

        # Get versioned output directories from file_manager (both full paths and relative paths)
        output_dirs_full = get_output_dirs(config, location_spec, omit_base_path=False)

        # Map data_level to output_dir key
        data_level_to_dir_key = {
            "a1_std": "standardized",
            "a2_std_partition": "standardized_partition",
            "b1_vap": "vap",
            "b1_vap_daily_compressed": "vap_daily_compressed",
            "b2_monthly_mean_vap": "monthly_summary_vap",
            "b3_yearly_mean_vap": "yearly_summary_vap",
            "b1_vap_by_point_partition": "vap_partition",
            "b4_vap_summary_parquet": "vap_summary_parquet",
            "b5_vap_atlas_summary_parquet": "vap_atlas_summary_parquet",
            "hsds": "hsds",
        }

        for data_level in location_levels:
            # Handle 00_raw separately (uses input directory structure)
            if data_level == "00_raw":
                base_path = Path(config["dir"]["base"])
                input_dir_template = config["dir"]["input"]["original"]
                input_dir = input_dir_template.replace(
                    "<location>", location_spec["output_name"]
                )
                local_dir = base_path / input_dir
            else:
                # Get directory key
                dir_key = data_level_to_dir_key.get(data_level)
                if not dir_key:
                    print(
                        f"Error: No directory mapping for data level '{data_level}'",
                        file=sys.stderr,
                    )
                    sys.exit(1)

                local_dir = output_dirs_full[dir_key]

            if not local_dir.exists():
                print(
                    f"Warning: Directory does not exist: {local_dir}", file=sys.stderr
                )
                files_by_level[data_level] = []
                continue

            # Get file discovery configuration
            file_config = DATA_LEVEL_CONFIG[data_level]
            valid_extensions = file_config["valid_extensions"]
            should_recurse = file_config["should_recurse"]

            # Discover files based on recursion and extension configuration
            files = []

            if should_recurse:
                # Recursive search
                for file_path in local_dir.rglob("*"):
                    if file_path.is_file():
                        # Filter by extension if specified
                        if (
                            valid_extensions is None
                            or file_path.suffix in valid_extensions
                        ):
                            relative_path = file_path.relative_to(local_dir)
                            files.append((str(file_path), str(relative_path)))
            else:
                # Non-recursive search (only immediate directory)
                for file_path in local_dir.glob("*"):
                    if file_path.is_file():
                        # Filter by extension if specified
                        if (
                            valid_extensions is None
                            or file_path.suffix in valid_extensions
                        ):
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


def get_data_level_paths(location_key, data_level):
    """
    Get local directory and S3 base path for a data level.

    Args:
        location_key: Location key (e.g., 'cook_inlet') - ignored for global data levels
        data_level: Data level directory name (e.g., 'b1_vap', 'manifest')

    Returns:
        Tuple of (local_dir, s3_base_path)
    """
    # Handle global data levels (like manifest)
    if DATA_LEVEL_CONFIG.get(data_level, {}).get("is_global", False):
        if data_level == "manifest":
            manifest_version = config["manifest"]["version"]
            local_dir = get_manifest_output_dir(config, manifest_version)
            s3_base_path = f"manifest/v{manifest_version}"
            return local_dir, s3_base_path

    # Handle location-specific data levels
    location_spec = config["location_specification"][location_key]
    output_dirs_full = get_output_dirs(config, location_spec, omit_base_path=False)
    output_dirs_relative = get_output_dirs(config, location_spec, omit_base_path=True)

    # Map data_level to output_dir key
    data_level_to_dir_key = {
        "a1_std": "standardized",
        "a2_std_partition": "standardized_partition",
        "b1_vap": "vap",
        "b1_vap_daily_compressed": "vap_daily_compressed",
        "b2_monthly_mean_vap": "monthly_summary_vap",
        "b3_yearly_mean_vap": "yearly_summary_vap",
        "b1_vap_by_point_partition": "vap_partition",
        "b4_vap_summary_parquet": "vap_summary_parquet",
        "b5_vap_atlas_summary_parquet": "vap_atlas_summary_parquet",
        "hsds": "hsds",
    }

    # Handle 00_raw separately
    if data_level == "00_raw":
        base_path = Path(config["dir"]["base"])
        input_dir_template = config["dir"]["input"]["original"]
        input_dir = input_dir_template.replace(
            "<location>", location_spec["output_name"]
        )
        local_dir = base_path / input_dir
        s3_base_path = str(
            Path(location_spec["output_name"])
            / f"v{config['dataset']['version']}"
            / data_level
        )
    else:
        dir_key = data_level_to_dir_key.get(data_level)
        local_dir = output_dirs_full[dir_key]
        s3_base_path = str(output_dirs_relative[dir_key])

    return local_dir, s3_base_path


def construct_s3_destination(location_key, data_level, relative_path):
    """
    Construct S3 destination path using file_manager for consistent versioning.

    Format for location-specific levels: <location>/v<version>/<data_level>/<relative_path>
    Format for global levels (manifest): manifest/v<manifest_version>/<relative_path>

    Args:
        location_key: Location key (e.g., 'cook_inlet') - ignored for global data levels
        data_level: Data level directory name (e.g., 'b1_vap', 'manifest')
        relative_path: Relative path within data level (e.g., '2005/file.nc')

    Returns:
        S3 destination path
    """
    # Handle global data levels (like manifest)
    if DATA_LEVEL_CONFIG.get(data_level, {}).get("is_global", False):
        if data_level == "manifest":
            manifest_version = config["manifest"]["version"]
            s3_base_relative_path = Path("manifest") / f"v{manifest_version}"
            return str(s3_base_relative_path / relative_path)

    # Handle location-specific data levels
    location_spec = config["location_specification"][location_key]
    output_dirs_relative = get_output_dirs(config, location_spec, omit_base_path=True)

    # Map data_level to output_dir key
    data_level_to_dir_key = {
        "a1_std": "standardized",
        "a2_std_partition": "standardized_partition",
        "b1_vap": "vap",
        "b1_vap_daily_compressed": "vap_daily_compressed",
        "b2_monthly_mean_vap": "monthly_summary_vap",
        "b3_yearly_mean_vap": "yearly_summary_vap",
        "b1_vap_by_point_partition": "vap_partition",
        "b4_vap_summary_parquet": "vap_summary_parquet",
        "b5_vap_atlas_summary_parquet": "vap_atlas_summary_parquet",
        "hsds": "hsds",
    }

    # Handle 00_raw separately
    if data_level == "00_raw":
        s3_base_relative_path = (
            Path(location_spec["output_name"])
            / f"v{config['dataset']['version']}"
            / data_level
        )
    else:
        dir_key = data_level_to_dir_key.get(data_level)
        s3_base_relative_path = output_dirs_relative[dir_key]

    return str(s3_base_relative_path / relative_path)


def display_summary(location_key, files_by_level):
    """Display summary of files to be uploaded."""
    # Check if we have any location-specific levels
    has_location_levels = any(
        not DATA_LEVEL_CONFIG.get(level, {}).get("is_global", False)
        for level in files_by_level.keys()
    )
    has_global_levels = any(
        DATA_LEVEL_CONFIG.get(level, {}).get("is_global", False)
        for level in files_by_level.keys()
    )

    version = config["dataset"]["version"]
    manifest_version = config["manifest"]["version"]
    total_files = sum(len(files) for files in files_by_level.values())
    total_size = calculate_total_size(files_by_level)

    # Use compact mode for large file counts
    compact_mode = total_files > 500

    print("\n" + "=" * 60)
    print("S3 UPLOAD SUMMARY")
    print("=" * 60)

    if has_location_levels:
        output_name = LOCATION_MAP.get(location_key, location_key)
        print(f"Location:     {location_key} ({output_name})")
        print(f"Version:      {version}")

    if has_global_levels:
        print(f"Manifest:     v{manifest_version}")

    print(f"Total files:  {total_files:,}")
    print(f"Total size:   {get_file_size_human(total_size)}")
    print("Bucket:       s3://oedi-data-drop/us-tidal")

    print("\nData Levels:")
    for data_level, files in files_by_level.items():
        level_size = sum(
            os.path.getsize(local_path)
            for local_path, _ in files
            if os.path.exists(local_path)
        )
        config_info = DATA_LEVEL_CONFIG[data_level]
        is_global = config_info.get("is_global", False)
        global_marker = " [GLOBAL]" if is_global else ""

        # Calculate percentage of total
        pct = (len(files) / total_files * 100) if total_files > 0 else 0

        if compact_mode:
            print(
                f"  {data_level}: {len(files):,} files ({pct:.1f}%) - {get_file_size_human(level_size)}{global_marker}"
            )
        else:
            print(
                f"  {data_level:30s} {len(files):6d} files  "
                f"{get_file_size_human(level_size):>12s}{global_marker}"
            )

    # Show sample file paths (fewer for compact mode)
    sample_limit = 3 if compact_mode else 10
    print("\nSample destinations:")
    sample_count = 0
    for data_level, files in files_by_level.items():
        for local_path, relative_path in files[:sample_limit]:
            if sample_count >= sample_limit:
                break
            s3_dest = construct_s3_destination(location_key, data_level, relative_path)
            print(f"  .../{'/'.join(s3_dest.split('/')[-2:])}")
            sample_count += 1
        if sample_count >= sample_limit:
            break

    if total_files > sample_limit:
        print(f"  ... and {total_files - sample_limit:,} more")

    print("=" * 60)


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


def upload_file_direct(
    local_file,
    s3_destination,
    dry_run=False,
):
    """
    Upload a single file directly using aws s3 cp.

    Args:
        local_file: Full path to local file
        s3_destination: S3 destination path (relative, without bucket)
        dry_run: Print command without executing

    Returns:
        True if successful, False otherwise
    """
    s3_bucket = "oedi-data-drop"
    s3_prefix = "us-tidal"
    full_s3_path = f"s3://{s3_bucket}/{s3_prefix}/{s3_destination}"

    aws_args = [
        "aws",
        "s3",
        "cp",
        "--profile",
        AWS_PROFILE,
        local_file,
        full_s3_path,
    ]

    if dry_run:
        print(f"[DRY RUN] {' '.join(aws_args)}")
        return True

    try:
        subprocess.run(aws_args, capture_output=True, text=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error uploading {local_file}: {e}", file=sys.stderr)
        if e.stderr:
            print(f"stderr: {e.stderr}", file=sys.stderr)
        return False


def upload_directory_direct(
    local_dir,
    s3_destination,
    file_count=0,
    dry_run=False,
):
    """
    Upload an entire directory recursively using aws s3 cp --recursive.

    Args:
        local_dir: Full path to local directory
        s3_destination: S3 destination path (relative, without bucket)
        file_count: Number of files being uploaded (for display)
        dry_run: Print command without executing

    Returns:
        True if successful, False otherwise
    """
    s3_bucket = "oedi-data-drop"
    s3_prefix = "us-tidal"
    full_s3_path = f"s3://{s3_bucket}/{s3_prefix}/{s3_destination}"

    aws_args = [
        "aws",
        "s3",
        "cp",
        "--profile",
        AWS_PROFILE,
        "--recursive",
        "--only-show-errors",
        str(local_dir),
        full_s3_path,
    ]

    if dry_run:
        print(f"[DRY RUN] {' '.join(aws_args)}")
        return True

    # Compact output
    print(f"  {local_dir} -> {full_s3_path}")
    if file_count > 0:
        print(f"  Uploading {file_count:,} files...", end=" ", flush=True)
    else:
        print("  Uploading...", end=" ", flush=True)

    try:
        subprocess.run(aws_args, check=True)
        print("Done.")
        return True
    except subprocess.CalledProcessError as e:
        print("FAILED!")
        print(f"  Error: {e}", file=sys.stderr)
        return False


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
  Location-specific:
    00_raw                      - Raw data (6h time limit)
    a1_std                      - Standardized data (6h time limit)
    a2_std_partition            - Standardized partition (6h time limit)
    b1_vap                      - Value-added products (6h time limit)
    b1_vap_daily_compressed     - Compressed VAP daily (6h time limit)
    b2_monthly_mean_vap         - Monthly mean VAP (6h time limit)
    b3_yearly_mean_vap          - Yearly mean VAP (6h time limit)
    b1_vap_by_point_partition            - VAP partition (6h time limit)
    b4_vap_summary_parquet      - VAP summary parquet (6h time limit)
    b5_vap_atlas_summary_parquet - VAP atlas summary parquet (6h time limit)
    hsds                        - HSDS format (24h time limit)

  Global (not location-specific):
    manifest                    - Global manifest JSON files (1h time limit)

Examples:
  # Upload all b1_vap files for Cook Inlet
  python dispatch_s3_upload.py cook_inlet --data-levels b1_vap

  # Upload multiple data levels
  python dispatch_s3_upload.py puget_sound --data-levels b1_vap hsds

  # Upload global manifest (location is ignored for global data levels)
  python dispatch_s3_upload.py _ --data-levels manifest

  # Upload manifest along with location-specific data
  python dispatch_s3_upload.py cook_inlet --data-levels b1_vap manifest

  # Dry run (don't actually submit jobs)
  python dispatch_s3_upload.py cook_inlet --data-levels b1_vap --dry-run

  # Skip checksum verification (faster)
  python dispatch_s3_upload.py cook_inlet --data-levels b1_vap --no-verify-checksum

  # Force upload (overwrite existing files)
  python dispatch_s3_upload.py cook_inlet --data-levels b1_vap --no-skip-if-exists

  # Direct upload (bypass SLURM, use aws s3 cp directly)
  python dispatch_s3_upload.py _ --data-levels manifest --direct

  # Direct upload with auto-confirm
  python dispatch_s3_upload.py _ --data-levels manifest --direct -y
        """,
    )

    parser.add_argument(
        "location",
        help="Location key (e.g., cook_inlet, puget_sound). Use '_' for global-only data levels like 'manifest'.",
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
        help="Print actions without executing uploads",
    )

    parser.add_argument(
        "--direct",
        action="store_true",
        help="Upload directly using 'aws s3 cp' instead of SLURM jobs (useful when SLURM is down)",
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

    # Show upload mode
    upload_mode = "DIRECT (aws s3 cp)" if args.direct else "SLURM jobs"
    print(f"\nUpload mode: {upload_mode}")

    # Confirm upload (unless --yes or --dry-run)
    if not args.yes and not args.dry_run:
        if not confirm_upload():
            print("Upload cancelled.")
            sys.exit(0)

    if args.direct:
        # Direct upload mode using aws s3 cp --recursive (directory-based)
        print("\nUploading directories directly...")
        successful_levels = 0
        failed_levels = 0

        for data_level, files in files_by_level.items():
            if not files:
                continue

            # Get the local directory and S3 base path for this data level
            local_dir, s3_base_path = get_data_level_paths(args.location, data_level)

            print(f"\n{data_level}:")

            success = upload_directory_direct(
                local_dir=local_dir,
                s3_destination=s3_base_path,
                file_count=len(files),
                dry_run=args.dry_run,
            )

            if success:
                successful_levels += 1
            else:
                failed_levels += 1

        # Summary for direct mode
        print("\n" + "=" * 80)
        if args.dry_run:
            print(
                f"DRY RUN COMPLETE - Would have uploaded {len(files_by_level)} directories ({total_files} files)"
            )
        else:
            print(
                f"Upload complete: {successful_levels} directories succeeded, {failed_levels} failed"
            )
        print("=" * 80)

    else:
        # SLURM job submission mode
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

        # Summary for SLURM mode
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
