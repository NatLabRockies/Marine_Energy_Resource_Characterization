#!/usr/bin/env python3
"""
Adhoc script to fix b1_vap filenames that have duplicate year format.

This script renames files from the incorrect format:
    AK_cook_inlet.wpto_high_res_tidal.b1_vap.20050101.20050101.v1.0.0.nc
To the correct format:
    AK_cook_inlet.wpto_high_res_tidal.b1_vap.20050101.000000.v1.0.0.nc

The second timestamp should be the time (HHMMSS) of the first timestamp in the file,
not the date of the last timestamp.

Usage:
    python fix_b1_vap_filenames.py <directory> [--dry-run]

Example:
    # Preview changes without renaming
    python fix_b1_vap_filenames.py /path/to/b1_vap_files --dry-run

    # Actually rename the files
    python fix_b1_vap_filenames.py /path/to/b1_vap_files
"""

import argparse
import re
from datetime import datetime
from pathlib import Path

import h5py
import pandas as pd


def extract_first_timestamp_from_file(nc_file):
    """
    Open netCDF file and extract the first timestamp using h5py.

    Uses h5py for fast, memory-efficient reading of just the first time value.
    Time is stored as Unix seconds (seconds since 1970-01-01).

    Args:
        nc_file: Path to netCDF file

    Returns:
        tuple: (date_str, time_str) in format (YYYYMMDD, HHMMSS)
    """
    try:
        with h5py.File(nc_file, "r") as f:
            # Read only the first time value (Unix seconds)
            first_time_unix = f["time"][0]
            print(f"  First time (Unix seconds): {first_time_unix}")

            # Convert from Unix seconds to datetime
            first_time = pd.to_datetime(
                first_time_unix, unit="s", origin="unix", utc=True
            )

            date_str = first_time.strftime("%Y%m%d")
            time_str = first_time.strftime("%H%M%S")

            print(f"Converted first timestamp: {date_str} {time_str}")

            return date_str, time_str
    except Exception as e:
        print(f"Error reading {nc_file.name}: {e}")
        return None, None


def parse_b1_vap_filename(filename):
    """
    Parse a b1_vap filename to extract components.

    Expected format:
        {output_name}.{dataset_name}.b1_vap.{temporal_start}.{temporal_end}.v{version}.nc

    Args:
        filename: Filename string

    Returns:
        dict: Parsed components or None if pattern doesn't match
    """
    # Pattern matches: <prefix>.b1_vap.<8digits>.<8digits>.v<version>.nc
    pattern = r"^(.+)\.b1_vap\.(\d{8})\.(\d{8})\.(v[\d\.]+)\.nc$"
    match = re.match(pattern, filename)

    if match:
        return {
            "prefix": match.group(1),
            "temporal_start": match.group(2),
            "temporal_end": match.group(3),
            "version": match.group(4),
        }
    return None


def is_incorrect_filename(parsed):
    """
    Check if filename has the duplicate year bug.

    The bug occurs when temporal_end is also a date (YYYYMMDD) instead of time (HHMMSS).
    We check if temporal_end looks like a date (first 4 digits >= 1900).

    Args:
        parsed: Parsed filename dict

    Returns:
        bool: True if filename needs fixing
    """
    if parsed is None:
        return False

    temporal_end = parsed["temporal_end"]

    # If temporal_end starts with 19xx or 20xx, it's likely a date not a time
    year_prefix = int(temporal_end[:4])
    return year_prefix >= 1900


def generate_correct_filename(parsed, time_str):
    """
    Generate corrected filename.

    Args:
        parsed: Parsed filename dict
        time_str: Correct time string (HHMMSS)

    Returns:
        str: Corrected filename
    """
    return f"{parsed['prefix']}.b1_vap.{parsed['temporal_start']}.{time_str}.{parsed['version']}.nc"


def process_directory(directory, dry_run=True):
    """
    Process all b1_vap files in directory and rename as needed.

    Args:
        directory: Path to directory containing b1_vap files
        dry_run: If True, only preview changes without renaming
    """
    dir_path = Path(directory)

    if not dir_path.exists():
        print(f"Error: Directory does not exist: {directory}")
        return

    if not dir_path.is_dir():
        print(f"Error: Not a directory: {directory}")
        return

    # Find all .nc files
    nc_files = sorted(dir_path.glob("*.nc"))

    if not nc_files:
        print(f"No .nc files found in {directory}")
        return

    print(f"Found {len(nc_files)} .nc files")
    print(f"Mode: {'DRY RUN (no changes)' if dry_run else 'RENAMING FILES'}")
    print("-" * 80)

    files_to_rename = []
    files_skipped = []

    for nc_file in nc_files:
        filename = nc_file.name
        parsed = parse_b1_vap_filename(filename)

        if parsed is None:
            files_skipped.append((filename, "Doesn't match b1_vap pattern"))
            continue

        if not is_incorrect_filename(parsed):
            files_skipped.append((filename, "Already correct format"))
            continue

        print("Extracting timestamp from file:", filename)
        # Extract actual first timestamp from file
        date_str, time_str = extract_first_timestamp_from_file(nc_file)

        if date_str is None or time_str is None:
            files_skipped.append((filename, "Could not read timestamp"))
            continue

        # Verify date matches
        if date_str != parsed["temporal_start"]:
            print(f"WARNING: Date mismatch for {filename}")
            print(f"  Filename has: {parsed['temporal_start']}")
            print(f"  File data has: {date_str}")
            files_skipped.append((filename, "Date mismatch"))
            continue

        new_filename = generate_correct_filename(parsed, time_str)
        files_to_rename.append((nc_file, new_filename))

    # Print summary
    print(f"\nFiles to rename: {len(files_to_rename)}")
    print(f"Files skipped: {len(files_skipped)}")
    print()

    # Print rename operations
    if files_to_rename:
        print("Rename operations:")
        print("-" * 80)
        for old_path, new_name in files_to_rename:
            print(f"OLD: {old_path.name}")
            print(f"NEW: {new_name}")
            print()

    # Print skipped files if verbose
    if files_skipped:
        print("\nSkipped files:")
        print("-" * 80)
        for filename, reason in files_skipped:
            print(f"{filename}: {reason}")
        print()

    # Perform renames if not dry run
    if not dry_run and files_to_rename:
        print("Performing renames...")
        print("-" * 80)
        success_count = 0
        error_count = 0

        for old_path, new_name in files_to_rename:
            new_path = old_path.parent / new_name

            if new_path.exists():
                print(f"ERROR: Target file already exists: {new_name}")
                error_count += 1
                continue

            try:
                old_path.rename(new_path)
                print(f"✓ Renamed: {old_path.name} -> {new_name}")
                success_count += 1
            except Exception as e:
                print(f"✗ Error renaming {old_path.name}: {e}")
                error_count += 1

        print()
        print(f"Successfully renamed: {success_count} files")
        if error_count > 0:
            print(f"Errors: {error_count} files")

    elif dry_run:
        print("\nDRY RUN: No files were renamed.")
        print("Run without --dry-run to perform the renames.")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Fix b1_vap filenames with duplicate year format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preview changes without renaming
  python fix_b1_vap_filenames.py /path/to/b1_vap_files --dry-run

  # Actually rename the files
  python fix_b1_vap_filenames.py /path/to/b1_vap_files
        """,
    )

    parser.add_argument(
        "directory",
        help="Directory containing b1_vap .nc files to fix",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without actually renaming files (default: False)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_directory(args.directory, dry_run=args.dry_run)
