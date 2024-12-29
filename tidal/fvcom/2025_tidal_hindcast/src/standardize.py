# Create one giant dataset using the valid_timestamps
import os

from pathlib import Path


import pandas as pd
import xarray as xr

from . import file_manager
from . import time_manager

from . import version


def remove_non_user_centric_variables(ds, config):
    variable_spec = config["model_specification"]
    user_centric_variable_names = list(variable_spec.keys())

    # Get variables to drop - those in dataset but not in specification
    variables_to_drop = [
        var for var in ds.variables if var not in user_centric_variable_names
    ]

    # Remove coordinate variables from the drop list to preserve dataset structure
    coord_vars = list(ds.coords)
    variables_to_drop = [var for var in variables_to_drop if var not in coord_vars]

    # Drop non-user variables while preserving coordinates
    return ds.drop_vars(variables_to_drop)


def standardize_time(ds, pandas_utc_timestamps):
    new_time = xr.DataArray(
        pandas_utc_timestamps,
        dims="time",
        name="time",
        attrs={
            "long_name": "Time",
            "standard_name": "time",
            "timezone": "UTC",
            "data_source": "derived",
            "data_source_description": "Derived from `Times` variable string. Original data `time` is in non standard modified julian timestamps",
            # ValueError: failed to prevent overwriting existing key units in attrs on variable 'time'. This is probably an encoding field used by xarray to describe how a variable is serialized. To proceed, remove this key from the variable's attributes manually.
            # "units": "seconds since 1970-01-01 00:00:00",
        },
    )
    ds = ds.drop_vars("time")
    ds = ds.assign_coords(time=new_time)
    return ds


def standardize_coords(ds):
    return ds


def standardize_global_metadata(ds, config, source_files):
    # Store original metadata with prefix
    existing_metadata = {f"original_{key}": value for key, value in ds.attrs.items()}

    # Get common metadata from config
    new_common_metadata = config["metadata"]

    # Get software version
    software_version = version.version

    # Get processing timestamp and user info
    processing_time = pd.Timestamp.now(tz="UTC").isoformat()
    processing_user = os.getenv("USER", "unknown")

    # Format source files information
    source_files_metadata = {
        "source_files": ", ".join(source_files),
        "source_file_count": len(source_files),
    }

    def get_conda_info():
        """Get Conda environment information including packages and version."""
        try:
            import subprocess
            import json

            # Get conda environment name
            env_name = os.getenv("CONDA_DEFAULT_ENV", "unknown")

            # Get conda version
            conda_version = subprocess.run(
                ["conda", "--version"], capture_output=True, text=True
            ).stdout.strip()

            # Get list of installed packages
            conda_list = subprocess.run(
                ["conda", "list", "--json"], capture_output=True, text=True
            )
            packages = json.loads(conda_list.stdout)

            # Format package list as a more concise string
            package_str = ", ".join(
                f"{pkg['name']}={pkg['version']}" for pkg in packages
            )

            return {
                "conda_environment": env_name,
                "conda_version": conda_version,
                "conda_packages": package_str,
            }
        except Exception as e:
            return {
                "conda_environment": "error_getting_conda_info",
                "conda_version": "error_getting_conda_info",
                "conda_packages": f"error_getting_conda_info: {str(e)}",
            }

    # Create complete new metadata dictionary
    new_metadata = {
        # Processing information
        "processing_timestamp": processing_time,
        "processing_user": processing_user,
        "software_version": software_version,
        # Source files information
        **source_files_metadata,
        # Conda information
        **get_conda_info(),
        # Add common metadata from config
        **new_common_metadata,
        # Add prefixed original metadata
        **existing_metadata,
    }

    # Completely replace dataset attributes
    ds.attrs = new_metadata

    return ds


def standardize_dataset(config, location, nc_files, valid_timestamps_df):
    # At this step the timestamps can have duplicates. This step uses the strategy defined
    # in the config to remove duplicate timestamps
    drop_duplicates_keep_strategy = config["time_specification"][
        "drop_duplicate_timestamps_keep_strategy"
    ]
    time_df = valid_timestamps_df.drop_duplicates(keep=drop_duplicates_keep_strategy)
    time_manager.does_time_match_specification(
        time_df["timestamp"], location["expected_delta_t_seconds"]
    )

    output_ds = None
    for source_file, this_df in time_df.groupby("source_file"):
        print(f"Processing file: {source_file}")
        print(f"Number of timestamps: {len(this_df)}")
        print(f"Start time: {this_df['timestamp'].iloc[0]}")
        print(f"End time: {this_df['timestamp'].iloc[-1]}")

        # Open dataset with times not decoded
        this_ds = xr.open_dataset(source_file, decode_times=False)

        # Select only the timestamps we want using the 'original' values from time_df
        this_ds = this_ds.sel(time=this_df["original"].values)

        # Standardize the time coordinate using the Timestamp values
        this_ds = standardize_time(this_ds, this_df["timestamp"].values)

        # Remove non-user-centric variables
        this_ds = remove_non_user_centric_variables(this_ds, config)

        # Combine with existing output dataset if it exists
        if output_ds is None:
            output_ds = this_ds
        else:
            output_ds = xr.concat([output_ds, this_ds], dim="time")

        output_ds = standardize_global_metadata(
            output_ds, config, [str(f) for f in time_df["source_files"].to_list()]
        )

        print(output_ds.info())
        exit()

    std_output_dir = file_manager.get_standardized_output_dir(config)
    output_ds.to_netcdf(Path(std_output_dir, f"{location['output_name']}_std.nc"))

    return output_ds
