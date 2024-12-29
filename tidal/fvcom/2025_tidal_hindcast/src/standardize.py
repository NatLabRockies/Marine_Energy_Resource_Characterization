# Create one giant dataset using the valid_timestamps
import xarray as xr

from . import time_manager


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


def standardize_dataset(config, location, nc_files, valid_timestamps_df):
    # At this step the timestamps can have duplicates. This step uses the strategy defined
    # in the config to remove duplicate timestamps
    drop_duplicates_keep_strategy = config["time_specification"][
        "drop_duplicate_timestamps_keep_strategy"
    ]
    time_df = valid_timestamps_df.drop_duplicates(keep=drop_duplicates_keep_strategy)
    time_manager.does_time_match_specification(
        time_df["Timestamp"], location["expected_delta_t_seconds"]
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

        print(output_ds.info())

    # Sort the final dataset by time
    output_ds = output_ds.sortby("time")

    return output_ds
