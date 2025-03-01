import gc
import multiprocessing as mp

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from . import attrs_manager, file_manager, file_name_convention_manager, time_manager


def process_single_period(period_data, config, location, output_dir, count):
    """Process a single time period and save the resulting dataset."""
    period_start, period_df = period_data

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
        ds = xr.open_dataset(std_file)

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

        ds.close()
        gc.collect()

    if len(datasets) > 0:
        # Concatenate datasets
        print(f"[{count}] Concatenating {len(datasets)} datasets...")
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

        print(f"[{count}] Saving partition file: {output_path}...")
        combined_ds.to_netcdf(output_path, encoding=config["dataset"]["encoding"])
        print(f"[{count}] Saved partition file: {output_path}!")

        # Cleanup
        combined_ds.close()
        for ds in datasets:
            ds.close()
        datasets.clear()
        gc.collect()

    datasets = []
    gc.collect()

    return str(output_path) if len(datasets) > 0 else None


def partition_by_time(config, location_key, time_df, force_reprocess=False):
    """
    Partitions time-series data into separate files based on configured frequency.
    Skips processing if the expected number of output files already exists.
    Uses multiprocessing to speed up the operation.
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

    # Determine the number of processes to use
    num_processes = min(mp.cpu_count(), len(process_args))

    num_processes = int(num_processes / 2)

    # Process the time periods in parallel
    with mp.Pool(num_processes) as pool:
        results = pool.starmap(
            process_single_period,
            [(args[0], args[1], args[2], args[3], args[4]) for args in process_args],
        )

    # Filter out None results and create a list of output files
    partition_files = [Path(path) for path in results if path is not None]

    print(
        f"Completed processing {len(partition_files)} time partitions with multiprocessing."
    )

    return partition_files
