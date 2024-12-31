from pathlib import Path
import gc

import numpy as np
import pandas as pd
import xarray as xr


# Chunk size corresponds to time, so chunk size of 4 means process 4 timestamps
def partition_by_time(config, location_key, time_df, freq="M", chunk_size=4):
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
        source_filenames = set()

        # Process each source file
        for std_file in unique_std_files:
            print(f"Adding {std_file} to {period} output dataset")
            # Open dataset with chunking
            ds = xr.open_dataset(std_file, chunks={"time": chunk_size})

            # Add source filenames
            if "source_files" in ds.attrs:
                filenames = [Path(f).name for f in ds.attrs["source_files"]]
                source_filenames.update(filenames)

            # Get timestamps for this file and period
            file_timestamps = period_df[period_df["std_files"] == std_file][
                "timestamp"
            ].values

            # Process data in chunks
            time_indices = np.isin(ds.time.values, file_timestamps)
            ds_subset = ds.isel(time=time_indices)

            if ds_subset.time.size > 0:
                # Load data into memory for this subset only
                ds_subset = ds_subset.compute()
                datasets.append(ds_subset)

            # Cleanup after processing each file
            ds.close()
            gc.collect()

        if len(datasets) > 0:
            # Concatenate datasets
            combined_ds = xr.concat(datasets, dim="time")
            combined_ds.attrs["source_files"] = list(source_filenames)

            # Create output filename
            if freq == "M":
                period_str = period.strftime("%Y%m")
            elif freq == "Y":
                period_str = period.strftime("%Y")
            else:
                period_str = str(period)

            output_path = output_dir / f"{location_name}_{period_str}_partition.nc"

            # Save to disk
            combined_ds.to_netcdf(output_path)
            partition_files.append(output_path)
            print(f"Saved partition file: {output_path}")

            # Cleanup
            combined_ds.close()
            for ds in datasets:
                ds.close()
            datasets.clear()
            gc.collect()

            # Additional cleanup after each period
            datasets = []
            gc.collect()

    return partition_files
