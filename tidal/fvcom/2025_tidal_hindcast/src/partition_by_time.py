from pathlib import Path
import gc
import numpy as np
import pandas as pd
import xarray as xr
from . import file_manager


def partition_by_time(config, location_key, time_df):
    location = config["location_specification"][location_key]
    output_dir = file_manager.get_standardized_partition_output_dir(config, location)
    output_dir.mkdir(parents=True, exist_ok=True)
    partition_files = []
    location_name = config["location_specification"][location_key]["output_name"]

    # use Pandas frequency string for grouping periods
    # See: https://pandas.pydata.org/docs/user_guide/timeseries.html#period-aliases
    # Common options: 'H' (hourly), '12H', 'D' (daily), 'W' (weekly), 'M' (monthly), 'Y' (yearly)
    freq = location["partition_frequency"]

    # Convert timestamps
    time_df["timestamp"] = pd.to_datetime(time_df["timestamp"])

    # Group by time frequency using Grouper
    for count, (period_start, period_df) in enumerate(
        time_df.groupby(pd.Grouper(key="timestamp", freq=freq)), 1
    ):
        if period_df.empty:
            continue

        print(f"Processing period: {period_start} with frequency string {freq}")

        # Get unique source files for this period
        unique_std_files = period_df["std_files"].unique()
        datasets = []
        source_filenames = set()

        # Process each source file
        for std_file in unique_std_files:
            print(f"Adding {std_file} to {period_start} output dataset")

            # Open dataset with chunking
            print("Opening dataset...")
            ds = xr.open_dataset(std_file)

            # Add source filenames
            if "source_files" in ds.attrs:
                filenames = [Path(f).name for f in ds.attrs["source_files"]]
                source_filenames.update(filenames)

            # Get timestamps for this file and period
            file_timestamps = period_df[period_df["std_files"] == std_file][
                "timestamp"
            ].values
            print("file_timestamps:", file_timestamps)

            # Process data in chunks
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
            combined_ds.attrs["source_files"] = list(source_filenames)

            # Create output filename using period start time and count
            start_date = period_start.strftime("%Y%m%d")
            num_timestamps = len(combined_ds.time)
            output_path = (
                output_dir
                / f"{count:02d}.{location_name}.start_{start_date}.n_timestamps_{num_timestamps:04d}.nc"
            )

            # Save to disk
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
