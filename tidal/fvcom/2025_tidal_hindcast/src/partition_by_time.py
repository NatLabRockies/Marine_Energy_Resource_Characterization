from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


# M is monthly
def partition_by_time(config, location_key, time_df, freq="M"):
    output_dir = Path(config["dir"]["output"]["standardized_partition"])
    output_dir.mkdir(parents=True, exist_ok=True)
    partition_files = []

    location_name = config["location_specification"][location_key]["output_name"]

    # Group by time frequency
    time_df["period"] = pd.to_datetime(time_df["timestamp"]).dt.to_period(freq)

    # Process each time period
    for period, period_df in time_df.groupby("period"):
        print(f"Processing period: {period}")

        # Get unique source files for this period
        unique_std_files = period_df["std_files"].unique()
        datasets = []

        # Collect all source filenames (not paths) for this period
        source_filenames = set()

        # Read each source file and select relevant timestamps
        for std_file in unique_std_files:
            ds = xr.open_dataset(std_file)

            # Add source filenames from this dataset's attributes
            if "source_files" in ds.attrs:
                # Extract just the filenames from the full paths
                filenames = [Path(f).name for f in ds.attrs["source_files"]]
                source_filenames.update(filenames)

            # Convert timestamps to numpy array for selection
            file_timestamps = period_df[period_df["std_files"] == std_file][
                "timestamp"
            ].values

            # Select timestamps for this period using isel with where condition
            time_indices = np.isin(ds.time.values, file_timestamps)
            ds_subset = ds.isel(time=time_indices)

            if not ds_subset.time.size == 0:  # Only append if we have data
                datasets.append(ds_subset)

        # Concatenate all datasets for this period
        if datasets:
            combined_ds = xr.concat(datasets, dim="time")

            # Update the source_files attribute with the collected filenames
            combined_ds.attrs["source_files"] = list(source_filenames)

            # Create output filename based on period
            if freq == "M":
                period_str = period.strftime("%Y%m")
            elif freq == "Y":
                period_str = period.strftime("%Y")
            else:
                period_str = str(period)

            output_path = output_dir / f"{location_name}_{period_str}_partition.nc"
            combined_ds.to_netcdf(output_path)
            partition_files.append(output_path)
            print(f"Saved partition file: {output_path}")

            # Close datasets to free memory
            for ds in datasets:
                ds.close()
            combined_ds.close()

    return partition_files
