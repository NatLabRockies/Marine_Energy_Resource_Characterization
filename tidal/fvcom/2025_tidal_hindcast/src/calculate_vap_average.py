from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from . import attrs_manager, file_manager, file_name_convention_manager


def verify_timestamps(nc_files, expected_timestamps, expected_delta_t_seconds):
    """Verify timestamp integrity across all files before processing."""

    total_timestamps = 0
    last_timestamp = None
    expected_diff = pd.Timedelta(seconds=expected_delta_t_seconds)

    for nc_file in nc_files:
        # Open dataset with minimal loading
        ds = xr.open_dataset(nc_file, decode_times=True)
        times = ds.time.values

        # Check count
        total_timestamps += len(times)

        # Check spacing within file
        time_diffs = pd.Series(times).diff()
        if not all(time_diffs[1:] == expected_diff):
            raise ValueError(f"Irregular timestamp spacing in {nc_file}")

        # Check continuity between files
        if last_timestamp is not None:
            gap = pd.Timestamp(times[0]) - last_timestamp
            if gap != expected_diff:
                raise ValueError(f"Gap between files: {nc_file}")

        last_timestamp = pd.Timestamp(times[-1])
        ds.close()

    if total_timestamps != expected_timestamps:
        raise ValueError(
            f"Expected {expected_timestamps} timestamps, found {total_timestamps}"
        )

    return True


def verify_constant_variables(ds1, ds2, constant_vars):
    """
    Verify that specified variables have the same values across datasets.

    Args:
        ds1: First dataset containing the variables
        ds2: Second dataset containing the variables
        constant_vars: List of variable names to verify

    Returns:
        dict: Dictionary of variables with comparison results
    """
    results = {}

    for var_name in constant_vars:
        if var_name in ds1 and var_name in ds2:
            # Check if the values are identical
            if np.array_equal(ds1[var_name].values, ds2[var_name].values):
                results[var_name] = True
            else:
                results[var_name] = False
        else:
            # Variable missing in one of the datasets
            results[var_name] = False

    return results


def calculate_vap_average(config, location):
    """
    Calculate average values across VAP NC files using rolling average computation.
    Only variables are averaged, preserving original dimensions and coordinates.
    Time dimension will use the first timestamp from the dataset.
    All variable attributes are preserved in the output dataset.

    Certain variables specified in constant_variables will not be averaged,
    but will be verified to be identical across all files.

    Args:
        config: Configuration dictionary
        location: Location dictionary containing site-specific parameters
    """
    # List of variables that should remain constant (not averaged)
    constant_variables = ["nv"]

    location = config["location_specification"][location]
    vap_path = file_manager.get_vap_output_dir(config, location)
    vap_nc_files = sorted(list(vap_path.rglob("*.nc")))
    if len(vap_nc_files) < 12:
        raise ValueError(
            f"Expecting at least 12 files in {vap_path}, found {len(vap_nc_files)}: {vap_nc_files}"
        )

    # Typical year has 365 days
    days_per_year = 365
    if location["output_name"] == "WA_puget_sound":
        # Puget sound is missing one day
        days_per_year = 364
    seconds_per_year = days_per_year * 24 * 60 * 60
    expected_timestamps = int(seconds_per_year / location["expected_delta_t_seconds"])

    print("Verifying timestamps across all files...")
    verify_timestamps(
        vap_nc_files, expected_timestamps, location["expected_delta_t_seconds"]
    )

    # Initialize template using the first file
    print(f"Starting averaging of {len(vap_nc_files)} vap files...")
    running_avg = None
    count = 0
    source_files = []
    first_timestamp = None
    time_attrs = None  # Store time attributes
    var_attrs = {}  # Dictionary to store variable attributes
    first_ds = None  # Store the first dataset for constant variable verification

    # Process each file
    for i, nc_file in enumerate(vap_nc_files):
        print(f"Processing File {i}: {nc_file}")
        ds = xr.open_dataset(nc_file)
        current_times = len(ds.time)

        # Initialize running average and save first timestamp if needed
        if running_avg is None:
            running_avg = ds.isel(time=0).copy(deep=True)
            first_timestamp = ds.time.isel(time=0).values
            time_attrs = ds.time.attrs.copy()  # Save time attributes
            first_ds = (
                ds.copy()
            )  # Store first dataset for constant variable verification

            # Store all variable attributes
            for var in ds.data_vars:
                var_attrs[var] = ds[var].attrs.copy()

            # Only initialize variables that need averaging
            for var in running_avg.data_vars:
                if "time" in ds[var].dims and var not in constant_variables:
                    # Use float64 for better numerical stability
                    if np.issubdtype(running_avg[var].dtype, np.integer):
                        running_avg[var] = running_avg[var].astype(np.float64)
                    elif np.issubdtype(running_avg[var].dtype, np.floating):
                        running_avg[var] = running_avg[var].astype(np.float64)

                    # Initialize to zero (will be properly set in the first iteration)
                    running_avg[var].values.fill(0)
        else:
            # Verify constant variables
            for var in constant_variables:
                if var in list(ds.keys()) and var in list(first_ds.keys()):
                    # Check if values are identical
                    if not np.array_equal(ds[var].values, first_ds[var].values):
                        print(
                            f"WARNING: Variable '{var}' differs between files. Using value from first file."
                        )

        # Update running average for variables - using rolling average formula
        for var in ds.data_vars:
            if "time" in ds[var].dims and var not in constant_variables:
                # Calculate mean for this file
                if np.issubdtype(ds[var].dtype, np.integer):
                    # Safely convert integers to float64 first
                    file_avg = ds[var].astype(np.float64).mean(dim="time")
                else:
                    file_avg = ds[var].mean(dim="time")

                # Update the rolling average using the formula:
                # new_avg = old_avg + (new_value - old_avg) / new_count
                if count == 0:
                    # First iteration - simply set to the file average
                    running_avg[var] = file_avg
                else:
                    weight = current_times / (count + current_times)
                    running_avg[var] = (
                        running_avg[var] + (file_avg - running_avg[var]) * weight
                    )

        count += current_times
        source_files.append(str(nc_file))
        ds.close()

    print("Completing final average calculation...")
    # The running_avg already contains the final average values

    # Restore variable attributes and potentially convert back to original types
    for var in running_avg.data_vars:
        if (
            var not in running_avg.dims
            and var not in running_avg.coords
            and var not in constant_variables
        ):
            # Restore variable attributes
            if var in var_attrs:
                running_avg[var].attrs = var_attrs[var]

                # Optionally convert back to original type if safe and configured
                original_dtype = var_attrs[var].get("_original_dtype")
                if original_dtype and config.get("preserve_dtypes", False):
                    try:
                        # Test conversion for safety
                        test_conversion = running_avg[var].values.astype(original_dtype)
                        running_avg[var] = running_avg[var].astype(original_dtype)
                    except (OverflowError, ValueError):
                        print(
                            f"WARNING: Cannot safely convert {var} back to {original_dtype}. Keeping higher precision."
                        )

    # Restore the time coordinate with its attributes
    running_avg = running_avg.assign_coords(time=("time", [first_timestamp]))
    running_avg["time"].attrs = time_attrs  # Restore time attributes

    # Set up output dataset
    averaged_ds = running_avg

    print(averaged_ds.info())
    print(averaged_ds.time)

    print("Generating data level file name...")
    # Generate output filename
    data_level_file_name = (
        file_name_convention_manager.generate_filename_for_data_level(
            averaged_ds,
            location["output_name"],
            config["dataset"]["name"],
            "b2",
            temporal="1_year_average",
        )
    )

    print("Adding standard attributes...")
    # Add standard attributes
    averaged_ds = attrs_manager.standardize_dataset_global_attrs(
        averaged_ds,
        config,
        location,
        "b2",
        source_files,
    )

    output_path = Path(
        file_manager.get_summary_vap_output_dir(config, location),
        f"001.{data_level_file_name}",
    )
    print(f"\tSaving to {output_path}...")
    averaged_ds.to_netcdf(output_path, encoding=config["dataset"]["encoding"])
    averaged_ds.close()
    return output_path
