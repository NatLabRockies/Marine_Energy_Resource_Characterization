import concurrent.futures
import gc
import os

from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import xarray as xr

from . import attrs_manager, file_manager, file_name_convention_manager, time_manager


def partition_by_time(config, location_key, time_df, force_reprocess=False):
    """
    Partitions time-series data into separate files based on configured frequency.
    Skips processing if the expected number of output files already exists.
    """
    location = config["location_specification"][location_key]
    output_dir = file_manager.get_standardized_partition_output_dir(config, location)
    output_dir.mkdir(parents=True, exist_ok=True)
    partition_files = []

    # Get partition frequency from config and ensure timestamps are datetime objects
    freq = location["partition_frequency"]
    time_df["timestamp"] = pd.to_datetime(time_df["timestamp"])

    # Check how many existing files we have for this period number
    existing_files = list(output_dir.glob("*.nc"))

    files_to_be_generated_count = len(
        time_df.groupby(pd.Grouper(key="timestamp", freq=freq))
    )

    # Skip if we already have the expected number of files and aren't forcing reprocess
    if len(existing_files) == files_to_be_generated_count and not force_reprocess:
        print(
            f"{output_dir} already has {len(existing_files)} files, skipping time partitioning..."
        )
        return existing_files

    # Process data in time-based groups
    for count, (period_start, period_df) in enumerate(
        time_df.groupby(pd.Grouper(key="timestamp", freq=freq)), 1
    ):
        if period_df.empty:
            continue

        print(f"Processing period: {period_start} with frequency string {freq}")

        # Count how many unique source files we have for this period
        unique_std_files = period_df["std_files"].unique()

        datasets = []
        source_filenames = set()

        for std_file in unique_std_files:
            print(f"Adding {std_file} to {period_start} output dataset")

            print("Opening dataset...")
            ds = xr.open_dataset(std_file)

            # Track source filenames
            if "source_files" in ds.attrs:
                filenames = [Path(f).name for f in ds.attrs["source_files"]]
                source_filenames.update(filenames)

            # Get timestamps for this file and period
            file_timestamps = period_df[period_df["std_files"] == std_file][
                "timestamp"
            ].values
            print("file_timestamps:", file_timestamps)

            # Subset the dataset
            time_indices = np.isin(ds.time.values, file_timestamps)
            print("Subsetting dataset by time_indicies...")
            ds_subset = ds.isel(time=time_indices)

            if ds_subset.time.size > 0:
                print("Appending dataset...")
                datasets.append(ds_subset)

            ds.close()
            gc.collect()

        if len(datasets) > 0:
            # Concatenate datasets
            print(f"Concatenating {len(datasets)} datasets...")
            combined_ds = xr.concat(datasets, dim="time")

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

            output_path = Path(output_dir, f"{count:03d}.{data_level_file_name}")

            print(f"Saving partition file: {output_path}...")
            combined_ds.to_netcdf(output_path, encoding=config["dataset"]["encoding"])
            partition_files.append(output_path)
            print(f"Saved partition file: {output_path}!")

            # Cleanup
            combined_ds.close()
            for ds in datasets:
                ds.close()
            datasets.clear()
            gc.collect()

        datasets = []
        gc.collect()

    return partition_files


def save_single_timestamp(ds_single_time, output_path, encoding):
    print(f"Saving single timestamp file: {output_path}")
    ds_single_time.to_netcdf(output_path, encoding=encoding)
    print(f"Saved single timestamp file: {output_path}")


def single_timestamp_partition(
    config, location_key, force_reprocess=False, max_workers=None
):
    """
    Partitions each input file into separate files with one timestamp per file.
    Processes files in order and verifies timestamps are in sequence.
    Uses parallel processing for file saving operations.

    Args:
        config: Configuration dictionary
        location_key: Key for the location in the config
        force_reprocess: Whether to force reprocessing if files already exist
        max_workers: Maximum number of parallel workers for file saving.
                     If None, automatically determines based on system resources.

    Returns:
        List of generated output file paths
    """
    # Determine optimal number of workers based on available resources
    if max_workers is None:
        try:
            # First, check for HPC environment variables
            if "SLURM_CPUS_PER_TASK" in os.environ:
                # SLURM environment
                max_workers = int(os.environ.get("SLURM_CPUS_PER_TASK"))
                print(
                    f"SLURM environment detected: using {max_workers} workers based on SLURM_CPUS_PER_TASK"
                )
            else:
                # Use psutil to determine available CPU cores and memory
                try:
                    # Get CPU count and system memory information
                    cpu_count = psutil.cpu_count(logical=False)  # Physical cores only
                    if cpu_count is None:
                        cpu_count = psutil.cpu_count(
                            logical=True
                        )  # Logical cores as fallback

                    # Get available memory in GB
                    mem_info = psutil.virtual_memory()
                    avail_mem_gb = mem_info.available / (1024**3)

                    # For I/O bound tasks, we can use more workers than CPU cores
                    # but we'll cap at 2x the core count to avoid system overload
                    io_optimal = min(cpu_count * 2, 32)  # Cap at 32 threads

                    # Estimate workers based on memory - each worker might need ~2GB
                    # This is an estimate and can be adjusted based on dataset sizes
                    mem_optimal = max(1, int(avail_mem_gb / 2))

                    # Choose the minimum of the two constraints
                    max_workers = min(io_optimal, mem_optimal)
                    print(
                        f"Auto-detected system resources: {cpu_count} CPU cores, {avail_mem_gb:.1f}GB available memory"
                    )
                    print(
                        f"Setting max_workers to {max_workers} based on system resources"
                    )

                except (ImportError, AttributeError):
                    # Fallback to a conservative number if psutil fails
                    max_workers = 4
                    print(
                        f"Unable to detect system resources, defaulting to {max_workers} workers"
                    )
        except Exception as e:
            # In case of any errors, default to a safe value
            max_workers = 4
            print(
                f"Error determining optimal worker count: {str(e)}. Defaulting to {max_workers} workers"
            )

    print(f"Max workers is: {max_workers}")

    location = config["location_specification"][location_key]
    output_dir = file_manager.get_standardized_partition_output_dir(config, location)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_directory = file_manager.get_standardized_output_dir(config, location)
    # Sort input files to ensure we process them in order
    sorted_input_files = sorted(list(input_directory.rglob("*.nc")))

    # Keep track of all timestamps to verify ordering at the end
    all_timestamps = []

    # List to store futures for parallel processing
    futures = []
    partition_files = []

    # Global counter for output files
    file_counter = 1
    timestamp_counter = 1

    # Create a thread pool executor
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for input_file in sorted_input_files:
            print(f"Processing file: {input_file}")

            # Open the dataset
            print(f"Reading: {input_file}")
            ds = xr.open_dataset(input_file)

            # Get all timestamps in this file
            timestamps = pd.to_datetime(ds.time.values)

            # Process each timestamp individually
            for timestamp in timestamps:
                print(f"Processing timestamp {timestamp_counter}: {timestamp}")

                # Record this timestamp
                all_timestamps.append(timestamp)

                # Subset to just this timestamp
                time_index = np.where(ds.time.values == np.datetime64(timestamp))[0][0]
                print(f"Selecting time index: {time_index}")
                ds_single_time = ds.isel(time=[time_index])

                # Format timestamp for filename
                timestamp_str = pd.to_datetime(timestamp).strftime("%Y%m%dT%H%M%S")

                print("Building output_file name...")
                # Generate output filename
                data_level_file_name = (
                    file_name_convention_manager.generate_filename_for_data_level(
                        ds_single_time,
                        location["output_name"],
                        config["dataset"]["name"],
                        "a2",
                    )
                )

                output_path = Path(
                    output_dir,
                    # f"{file_counter:05d}.{data_level_file_name}",
                    data_level_file_name,
                )

                # Create a copy of the dataset for the thread to work with
                # This avoids potential race conditions if the main thread closes the dataset
                ds_copy = ds_single_time.copy(deep=True)

                # Submit the save task to the executor
                future = executor.submit(
                    save_single_timestamp,
                    ds_copy,
                    output_path,
                    config["dataset"]["encoding"],
                )

                # Store the future and output path
                futures.append((future, output_path))

                # Increment file counter for next file
                file_counter += 1
                timestamp_counter += 1

            # Close the dataset
            ds.close()
            gc.collect()

            # Check if any futures have completed - collect their results
            completed = [f for f in futures if f[0].done()]
            for future, path in completed:
                try:
                    # Get the result (will raise an exception if the future failed)
                    result = future.result()
                    partition_files.append(result)
                except Exception as e:
                    print(f"Error in processing {path}: {str(e)}")
                futures.remove((future, path))

        # Wait for all remaining futures to complete
        print(
            f"Waiting for {len(futures)} remaining file save operations to complete..."
        )
        for future, path in futures:
            try:
                result = future.result()
                partition_files.append(result)
            except Exception as e:
                print(f"Error in processing {path}: {str(e)}")

    # Verify timestamps are in order
    if all_timestamps:
        for i in range(1, len(all_timestamps)):
            if all_timestamps[i] < all_timestamps[i - 1]:
                error_msg = f"Timestamps not in order: {all_timestamps[i-1]} followed by {all_timestamps[i]}"
                print(error_msg)
                raise ValueError(error_msg)

    print(
        f"Successfully processed {len(partition_files)} timestamps across {len(sorted_input_files)} files"
    )
    return partition_files
