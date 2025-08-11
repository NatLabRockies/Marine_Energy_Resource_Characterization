import gc
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from . import (
    attrs_manager,
    file_manager,
    file_name_convention_manager,
    nc_manager,
    time_manager,
)


def process_single_period(period_data, config, location, output_dir, count):
    """Process a single time period and save the resulting dataset."""
    period_start, period_df = period_data
    output_path = None

    if period_df.empty:
        return None

    print(f"[{count}] Processing period: {period_start}")

    # Count how many unique source files we have for this period
    unique_std_files = period_df["std_files"].unique()

    datasets = []
    source_filenames = set()

    for std_file in unique_std_files:
        print(f"[{count}] Adding {std_file} to {period_start} output dataset")

        print(f"[{count}] Opening dataset...")

        ds = nc_manager.nc_open(std_file, config)

        # Track source filenames
        if "source_files" in ds.attrs:
            filenames = [Path(f).name for f in ds.attrs["source_files"]]
            source_filenames.update(filenames)

        # Get timestamps for this file and period
        file_timestamps = period_df[period_df["std_files"] == std_file][
            "timestamp"
        ].values
        print(f"[{count}] file_timestamps:", file_timestamps)

        # Subset the dataset
        time_indices = np.isin(ds.time.values, file_timestamps)
        print(f"[{count}] Subsetting dataset by time_indicies...")
        ds_subset = ds.isel(time=time_indices)

        if ds_subset.time.size > 0:
            print(f"[{count}] Appending dataset...")
            datasets.append(ds_subset)

    if len(datasets) > 0:
        # Concatenate datasets
        print(f"[{count}] Concatenating {len(datasets)} datasets...")

        # combined_ds = xr.concat(datasets, dim="time")

        # Separate dataset into time-varying and time-invariant variables
        # Xarray adds a "time" dimension to some variables ('nv' not sure why?)
        # This causes a massive size increase in the output file size and is incorrect
        # To fix we concat time variables only and just add back the time-invariant variables
        time_varying_vars = []
        time_invariant_vars = {}

        first_ds = datasets[0]
        for var_name in first_ds.data_vars:
            if "time" in first_ds[var_name].dims:
                time_varying_vars.append(var_name)
            else:
                time_invariant_vars[var_name] = first_ds[var_name]

        # Concatenate only time-varying variables
        if time_varying_vars:
            datasets_time_vars = [ds[time_varying_vars] for ds in datasets]
            combined_ds = xr.concat(datasets_time_vars, dim="time")

            # Add back time-invariant variables
            for var_name, var_data in time_invariant_vars.items():
                combined_ds[var_name] = var_data
        else:
            raise ValueError("No time-varying variables found in concat datasets.")

        combined_ds = attrs_manager.standardize_dataset_global_attrs(
            combined_ds, config, location, "a2", [str(f) for f in source_filenames]
        )

        temporal_string = time_manager.generate_temporal_attrs(combined_ds)[
            "standard_name"
        ]

        # Generate output filename
        data_level_file_name = (
            file_name_convention_manager.generate_filename_for_data_level(
                combined_ds,
                location["output_name"],
                config["dataset"]["name"],
                "a2",
                temporal=temporal_string,
            )
        )

        # output_path = Path(output_dir, f"{count:03d}.{data_level_file_name}")
        output_path = Path(output_dir, data_level_file_name)

        print(f"[{count}] Saving partition file: {output_path}...")
        nc_manager.nc_write(combined_ds, output_path, config)

        # Verify the file exists before continuing
        wait_count = 0
        while not output_path.exists() and wait_count < 10:
            time.sleep(0.5)
            wait_count += 1

        if output_path.exists():
            print(f"[{count}] Saved partition file: {output_path}!")
        else:
            print(f"[{count}] Warning: Could not verify file was saved: {output_path}")

        # Close the combined dataset
        combined_ds.close()
        combined_ds = None

    # Return the output path only if it was successfully created
    return str(output_path) if output_path and output_path.exists() else None


def partition_by_time(
    config, location_key, time_df, force_reprocess=False, use_multiprocessing=False
):
    """
    Partitions time-series data into separate files based on configured frequency.
    Skips processing if the expected number of output files already exists.
    Can run either sequentially or with multiprocessing based on the flag.

    Args:
        config: Configuration dictionary
        location_key: Key to look up location in the configuration
        time_df: DataFrame containing timestamps and file information
        force_reprocess: If True, reprocess even if files exist
        use_multiprocessing: If True, use multiprocessing; otherwise process sequentially
    """
    location = config["location_specification"][location_key]
    output_dir = file_manager.get_standardized_partition_output_dir(config, location)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get partition frequency from config and ensure timestamps are datetime objects
    freq = location["partition_frequency"]
    time_df["timestamp"] = pd.to_datetime(time_df["timestamp"])

    # Check how many existing files we have for this period number
    existing_files = list(output_dir.glob("*.nc"))

    # Group data by time periods
    time_groups = time_df.groupby(pd.Grouper(key="timestamp", freq=freq))
    files_to_be_generated_count = len(time_groups)

    # Skip if we already have the expected number of files and aren't forcing reprocess
    if len(existing_files) == files_to_be_generated_count and not force_reprocess:
        print(
            f"{output_dir} already has {len(existing_files)} files, skipping time partitioning..."
        )
        return existing_files

    # Create a list of arguments for each period to be processed
    process_args = []
    for count, period_data in enumerate(time_groups, 1):
        process_args.append((period_data, config, location, output_dir, count))

    results = []

    if use_multiprocessing:
        # Multiprocessing approach
        # Limit to 2 processes to avoid excessive file operations
        num_processes = 2
        print(
            f"Processing {len(process_args)} time partitions with {num_processes} processes"
        )

        # Process the time periods in parallel, but one chunk at a time to control memory
        with mp.Pool(num_processes) as pool:
            # Process in smaller chunks to avoid loading too much at once
            chunk_size = 2  # Process only 2 files at a time
            for i in range(0, len(process_args), chunk_size):
                chunk = process_args[i : i + chunk_size]
                print(
                    f"Processing chunk {i // chunk_size + 1}/{(len(process_args) - 1) // chunk_size + 1}"
                )

                # Process this chunk
                chunk_results = pool.starmap(
                    process_single_period,
                    [(args[0], args[1], args[2], args[3], args[4]) for args in chunk],
                )

                results.extend(chunk_results)

                # Force garbage collection between chunks
                gc.collect()

        print("Completed processing time partitions with multiprocessing.")
    else:
        # Sequential processing approach
        print(f"Processing {len(process_args)} time partitions sequentially")

        for count, args in enumerate(process_args, 1):
            print(f"Processing partition {count}/{len(process_args)}")
            result = process_single_period(args[0], args[1], args[2], args[3], args[4])
            results.append(result)
            # Force garbage collection after each file
            gc.collect()

        print("Completed processing time partitions sequentially.")

    # Wait a moment to ensure all file operations are complete
    time.sleep(1)

    # Filter out None results and create a list of output files
    partition_files = [Path(path) for path in results if path is not None]

    # Verify all files exist
    for file_path in partition_files:
        if not file_path.exists():
            print(f"Warning: Expected output file does not exist: {file_path}")

    print(f"Successfully created {len(partition_files)} partition files.")

    return partition_files
