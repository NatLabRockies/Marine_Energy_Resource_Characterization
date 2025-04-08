from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from . import attrs_manager, file_manager, file_name_convention_manager, nc_manager


def principal_flow_directions(
    directions, direction_bin_width_degrees=1, excluded_angle_range=180
):
    """
    Find the two most prominent directional peaks in flow data.

    Parameters
    ----------
    directions: array-like
        Flow direction in degrees (0-360)
    direction_bin_width_degrees: float, optional
        Width of directional bins in degrees, default is 1
    excluded_angle_range: float, optional
        Range of angles to exclude around each detected peak when searching for the second peak.
        The exclusion is centered on the peak, extending excluded_angle_range/2 in each direction.
        Default is 180 degrees.

    Returns
    -------
    tuple(float, float)
        The two principal flow directions in degrees, with NaN for direction 2 if not found
    """
    # Filter out NaN values
    valid_directions = np.array(directions)[~np.isnan(np.array(directions))]

    if len(valid_directions) == 0:
        return np.nan, np.nan

    # Create histogram with the specified bin width
    n_bins = int(360 / direction_bin_width_degrees)
    hist, bin_edges = np.histogram(valid_directions, bins=n_bins, range=[0, 360])
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Find the most prominent peak
    peak1_idx = np.argmax(hist)
    peak1_value = bin_centers[peak1_idx]

    # Calculate how many bins to exclude on each side of the peak
    half_window_bins = int((excluded_angle_range / 2) / direction_bin_width_degrees)

    # Create mask to exclude the first peak and a window around it
    mask = np.ones_like(hist, dtype=bool)
    for i in range(-half_window_bins, half_window_bins + 1):
        # Exclude around first peak
        idx1 = (peak1_idx + i) % n_bins
        mask[idx1] = False

        # Exclude around opposite direction
        idx2 = (peak1_idx + i + n_bins // 2) % n_bins
        mask[idx2] = False

    # Find the second peak in the remaining data
    if np.any(mask) and np.max(hist * mask) > 0:
        peak2_idx = np.argmax(hist * mask)
        peak2_value = bin_centers[peak2_idx]
    else:
        # If everything was excluded or no second peak, return NaN
        peak2_value = np.nan

    # Make first peak the smaller angle if both exist
    if not np.isnan(peak2_value) and peak1_value > peak2_value:
        peak1_value, peak2_value = peak2_value, peak1_value

    return peak1_value, peak2_value


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

    Additionally calculates principal flow directions across the full timeseries.

    Args:
        config: Configuration dictionary
        location: Location dictionary containing site-specific parameters
    """
    # List of variables that should remain constant (not averaged)
    constant_variables = ["nv"]

    # Flow direction variable name
    direction_var = "vap_sea_water_speed_to_direction"

    # Output variable names for principal directions
    principal_dir1_var = "vap_sea_water_primary_to_direction"
    principal_dir2_var = "vap_sea_water_secondary_to_direcion"

    # Configuration for principal flow direction calculation
    direction_bin_width_degrees = 1
    direction_excluded_angle_range = 180

    location = config["location_specification"][location]
    vap_path = file_manager.get_vap_output_dir(config, location)
    vap_nc_files = sorted(list(vap_path.rglob("*.nc")))
    if len(vap_nc_files) < 12:
        raise ValueError(
            f"Expecting at least 12 files in {vap_path}, found {len(vap_nc_files)}: {vap_nc_files}"
        )

    output_path = file_manager.get_summary_vap_output_dir(config, location)
    output_nc_files = list(output_path.rglob("*.nc"))

    if len(output_nc_files) > 0:
        print(
            f"{len(output_nc_files)} summary files already exist. Skipping calculate_vap_average"
        )
        return

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

    # Create a full timeseries xarray dataset to store directions
    # First, let's open the first file to get structure
    sample_ds = xr.open_dataset(vap_nc_files[0])

    # If direction variable exists, prepare a dataset to hold the full timeseries
    full_timeseries = None
    if direction_var in sample_ds:
        # Get the structure of the direction variable
        dims = sample_ds[direction_var].dims
        coords = {dim: sample_ds[dim] for dim in dims}

        # Initialize empty array - we'll fill this with all files
        full_timeseries = xr.Dataset()

        # Close sample dataset
        sample_ds.close()

        # Load the complete timeseries into memory
        print("Loading complete direction timeseries into memory...")
        all_ds = []
        for nc_file in vap_nc_files:
            ds = xr.open_dataset(nc_file)
            if direction_var in ds:
                all_ds.append(ds[[direction_var]])
            ds.close()

        if all_ds:
            full_timeseries = xr.concat(all_ds, dim="time")
            print(
                f"Loaded direction timeseries with shape: {full_timeseries[direction_var].shape}"
            )
    else:
        sample_ds.close()
        print(f"Warning: Direction variable '{direction_var}' not found in dataset")

    # Process each file for regular averaging
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

    # Calculate principal flow directions if we have direction data
    if full_timeseries is not None and direction_var in full_timeseries:
        print("Calculating principal flow directions...")
        # Get spatial dimensions (all dimensions except time)
        spatial_dims = [
            dim for dim in full_timeseries[direction_var].dims if dim != "time"
        ]

        # If there are spatial dimensions, we need to loop through each point
        if spatial_dims:
            # Create arrays to store the principal directions
            shape = tuple(len(full_timeseries[dim]) for dim in spatial_dims)
            principal_dir1_array = np.full(shape, np.nan)
            principal_dir2_array = np.full(shape, np.nan)

            # Use numpy's ndindex to iterate through all combinations of indices
            total_points = np.prod(shape)
            print(
                f"Processing principal directions for {total_points} spatial points..."
            )

            for idx_count, idx in enumerate(np.ndindex(shape)):
                if idx_count % 100 == 0:  # Progress update
                    print(f"  Processed {idx_count}/{total_points} points...")

                # Create the selection dictionary
                sel_dict = {
                    dim: full_timeseries[dim][i] for dim, i in zip(spatial_dims, idx)
                }

                # Extract direction timeseries at this point
                point_directions = full_timeseries[direction_var].sel(sel_dict).values

                # Calculate principal directions
                dir1, dir2 = principal_flow_directions(
                    point_directions,
                    direction_bin_width_degrees=direction_bin_width_degrees,
                    excluded_angle_range=direction_excluded_angle_range,
                )

                # Store results
                principal_dir1_array[idx] = dir1
                principal_dir2_array[idx] = dir2

            # Add to dataset with proper dimensions
            running_avg[principal_dir1_var] = (spatial_dims, principal_dir1_array)
            running_avg[principal_dir2_var] = (spatial_dims, principal_dir2_array)
        else:
            # Single point case - no spatial dimensions
            dir1, dir2 = principal_flow_directions(
                full_timeseries[direction_var].values,
                direction_bin_width_degrees=direction_bin_width_degrees,
                excluded_angle_range=direction_excluded_angle_range,
            )
            running_avg[principal_dir1_var] = ((), dir1)
            running_avg[principal_dir2_var] = ((), dir2)

        # Add CF-compliant attributes based on the to_direction attributes
        # Get original direction attributes if available
        dir_attrs = {}
        if direction_var in first_ds and hasattr(first_ds[direction_var], "attrs"):
            dir_attrs = first_ds[direction_var].attrs.copy()

        # Define principal direction 1 attributes
        running_avg[principal_dir1_var].attrs = {
            "standard_name": "sea_water_to_principal_direction_1",
            "long_name": "Primary principal direction of sea water flow",
            "units": "degree",
            "valid_min": 0.0,
            "valid_max": 360.0,
            "coverage_content_type": "physicalMeasurement",
            "computation": (
                "Principal direction calculated from histogram analysis of flow directions "
                f"using bin width of {direction_bin_width_degrees} degrees. "
                f"Represents the dominant flow direction over the entire timeseries."
            ),
            "direction_reference": dir_attrs.get(
                "direction_reference",
                "Direction in degrees clockwise from true north (0°)",
            ),
            "source_variable": direction_var,
            "method": (
                "Histogram analysis of to_direction values to identify primary peak. "
                f"Bin width: {direction_bin_width_degrees}°, Exclusion window: {direction_excluded_angle_range}°"
            ),
        }

        # Define principal direction 2 attributes
        running_avg[principal_dir2_var].attrs = {
            "standard_name": "sea_water_to_principal_direction_2",
            "long_name": "Secondary principal direction of sea water flow",
            "units": "degree",
            "valid_min": 0.0,
            "valid_max": 360.0,
            "coverage_content_type": "physicalMeasurement",
            "computation": (
                "Secondary principal direction calculated from histogram analysis of flow directions "
                f"using bin width of {direction_bin_width_degrees} degrees. "
                f"Represents the secondary dominant flow direction over the entire timeseries, "
                f"typically in the opposite direction of the primary flow."
            ),
            "direction_reference": dir_attrs.get(
                "direction_reference",
                "Direction in degrees clockwise from true north (0°)",
            ),
            "source_variable": direction_var,
            "method": (
                "Histogram analysis of to_direction values to identify secondary peak. "
                f"First peak and surrounding angles (±{direction_excluded_angle_range/2}°) are excluded. "
                f"Bin width: {direction_bin_width_degrees}°"
            ),
        }

        # Clean up full timeseries dataset to free memory
        del full_timeseries

    # Restore variable attributes and potentially convert back to original types
    for var in running_avg.data_vars:
        if (
            var not in running_avg.dims
            and var not in running_avg.coords
            and var not in constant_variables
            and var
            != principal_dir1_var  # Don't overwrite principal direction attributes
            and var
            != principal_dir2_var  # Don't overwrite principal direction attributes
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

    averaged_ds.to_netcdf(
        output_path,
        encoding=nc_manager.define_compression_encoding(
            averaged_ds,
            base_encoding=config["dataset"]["encoding"],
            compression_strategy="none",
        ),
    )
    return output_path
