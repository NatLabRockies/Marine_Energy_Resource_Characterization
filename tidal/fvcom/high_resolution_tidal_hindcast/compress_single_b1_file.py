"""
Worker script for compressing a single b1_vap file.

This script is called by SLURM array jobs. Each worker processes one b1_vap file
based on the file index (SLURM_ARRAY_TASK_ID).

Usage:
    python compress_single_b1_file.py <location> <file_index> [--skip-existing]

Example:
    python compress_single_b1_file.py cook_inlet 0
    python compress_single_b1_file.py aleutian_islands 5 --skip-existing
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import xarray as xr

from config import config
from src import file_manager, nc_manager


def validate_location(location):
    """Validate location argument."""
    if location not in config["location_specification"]:
        valid_locations = list(config["location_specification"].keys())
        raise argparse.ArgumentTypeError(
            f"Invalid location: {location}. Must be one of: {', '.join(valid_locations)}"
        )
    return location


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compress a single b1_vap file into daily compressed files."
    )

    parser.add_argument(
        "location",
        type=validate_location,
        help="Location to process (e.g., aleutian_islands, cook_inlet)",
    )

    parser.add_argument(
        "file_index",
        type=int,
        help="File index to process (0-based, from SLURM_ARRAY_TASK_ID)",
    )

    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip processing if output files already exist",
    )

    return parser.parse_args()


def get_b1_file_by_index(location_key, file_index):
    """
    Get b1_vap file by index using sorted glob.

    Args:
        location_key: Location key from config
        file_index: 0-based file index

    Returns:
        Path to b1_vap file

    Raises:
        IndexError: If file_index is out of range
    """
    location = config["location_specification"][location_key]
    vap_dir = file_manager.get_vap_output_dir(config, location)

    # Get sorted list of b1_vap files (must match dispatcher's sorting)
    b1_files = sorted(vap_dir.glob("*.nc"))

    if not b1_files:
        raise ValueError(f"No b1_vap files found in {vap_dir}")

    if file_index >= len(b1_files):
        raise IndexError(
            f"File index {file_index} out of range (0-{len(b1_files) - 1})"
        )

    return b1_files[file_index]


def process_single_b1_file(
    b1_file, output_dir, config, location, partition_freq="1D", skip_existing=False
):
    """
    Process a single b1_vap file and partition it into daily compressed files.

    Args:
        b1_file: Path to b1_vap file
        output_dir: Output directory for compressed files
        config: Configuration dictionary
        location: Location configuration
        partition_freq: Pandas frequency string for partitioning (default: "1D")
        skip_existing: If True, skip if output files already exist
    """
    print(f"Processing {b1_file.name}...")

    # Open the dataset
    ds = nc_manager.nc_open(b1_file, config)

    # Create a dataframe with time information
    time_df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(ds.time.values),
            "b1_file": str(b1_file),
        }
    )

    # Group by the partition frequency (1D for daily)
    time_groups = time_df.groupby(pd.Grouper(key="timestamp", freq=partition_freq))

    files_created = []
    files_skipped = []

    for period_start, period_df in time_groups:
        if period_df.empty:
            continue

        print(f"  Creating file for {period_start.date()}...")

        # Get timestamps for this period
        period_timestamps = period_df["timestamp"].values

        # Subset the dataset to this time period
        time_indices = pd.Index(ds.time.values).isin(period_timestamps)
        ds_subset = ds.isel(time=time_indices)

        if ds_subset.time.size == 0:
            continue

        # Generate output filename based on temporal coverage
        temporal_start = pd.Timestamp(ds_subset.time.values[0]).strftime("%Y%m%d")
        temporal_end = pd.Timestamp(ds_subset.time.values[-1]).strftime("%Y%m%d")

        output_name = location["output_name"]
        dataset_name = config["dataset"]["name"]
        version = config["dataset"]["version"]

        # Create filename following the naming convention
        filename = f"{output_name}.{dataset_name}.b1_vap.{temporal_start}.{temporal_end}.v{version}.nc"
        output_path = output_dir / filename

        # Skip if file exists and skip_existing is True
        if output_path.exists() and skip_existing:
            print(f"    File already exists (skipping): {output_path.name}")
            files_skipped.append(output_path)
            continue

        # Write the file with archival compression
        print(f"    Writing with archival compression: {output_path.name}")
        nc_manager.nc_write(
            ds_subset, output_path, config, compression_strategy="archival"
        )

        files_created.append(output_path)
        print(f"    Completed: {output_path.name}")

    # Close the dataset
    ds.close()

    return files_created, files_skipped


def main():
    """Main entry point."""
    args = parse_args()

    location_key = args.location
    file_index = args.file_index
    skip_existing = args.skip_existing

    print(f"Worker starting for {location_key}, file index {file_index}")
    print(f"Skip existing: {skip_existing}")
    print()

    # Get location configuration
    location = config["location_specification"][location_key]
    partition_freq = location.get("b1_archive_vap_partition", "1D")

    # Get the specific b1_vap file to process
    try:
        b1_file = get_b1_file_by_index(location_key, file_index)
    except (ValueError, IndexError) as e:
        print(f"Error: {e}")
        sys.exit(1)

    print(f"Target file: {b1_file}")
    print(f"Partition frequency: {partition_freq}")
    print()

    # Get output directory
    output_dir = file_manager.get_vap_daily_compressed_output_dir(config, location)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}")
    print()

    # Process the file
    files_created, files_skipped = process_single_b1_file(
        b1_file, output_dir, config, location, partition_freq, skip_existing
    )

    # Summary
    print()
    print("=" * 60)
    print("Processing Summary")
    print("=" * 60)
    print(f"Input file:      {b1_file.name}")
    print(f"Files created:   {len(files_created)}")
    print(f"Files skipped:   {len(files_skipped)}")
    print(f"Output location: {output_dir}")
    print("=" * 60)

    if files_created:
        print("\nCreated files:")
        for f in files_created:
            print(f"  - {f.name}")

    if files_skipped:
        print("\nSkipped files:")
        for f in files_skipped:
            print(f"  - {f.name}")

    print("\nWorker completed successfully!")


if __name__ == "__main__":
    main()
