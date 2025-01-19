from pathlib import Path
import gc
import numpy as np
import pandas as pd
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
            combined_ds.to_netcdf(output_path)
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
