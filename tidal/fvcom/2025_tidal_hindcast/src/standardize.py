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

    for _, this_df in time_df.group_by(["source_file"]):
        print(len(this_df))
        print(this_df["Timestamp"].iloc[0])
        print(this_df["Timestamp"].iloc[-1])
