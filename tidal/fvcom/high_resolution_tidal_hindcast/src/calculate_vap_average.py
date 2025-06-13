from pathlib import Path
from collections import defaultdict
import re


import numpy as np
import pandas as pd
import xarray as xr

from scipy.signal import find_peaks

from . import attrs_manager, file_manager, file_name_convention_manager, nc_manager


def verify_timestamps(nc_files, expected_timestamps, expected_delta_t_seconds, config):
    """Verify timestamp integrity across all files before processing."""

    total_timestamps = 0
    last_timestamp = None
    expected_diff = pd.Timedelta(seconds=expected_delta_t_seconds)

    for nc_file in nc_files:
        # Open dataset with minimal loading
        ds = nc_manager.nc_open(nc_file, config, decode_times=True)
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


def calculate_single_primary_and_secondary_direction(
    direction_timeseries, bin_width_degrees=1, mask_width_degrees=180
):
    """
    Calculate primary and secondary flow directions from a time series of directions.
    """
    min_degrees = 0
    max_degrees = 360
    n_bins = int(max_degrees / bin_width_degrees)
    dir_histogram, bin_edges = np.histogram(
        direction_timeseries,
        bins=n_bins,
        range=(min_degrees, max_degrees),
        density=True,
    )
    # Find primary direction (highest peak)
    primary_bin_idx = np.argmax(dir_histogram)
    primary_direction = bin_edges[primary_bin_idx]
    # Calculate all bin centers at once
    bin_centers = bin_edges[:-1]
    # Vectorized circular distance calculation
    angular_diff = np.minimum(
        np.abs(bin_centers - primary_direction),
        360 - np.abs(bin_centers - primary_direction),
    )
    # Create mask and apply it
    mask = angular_diff > mask_width_degrees / 2
    masked_histogram = dir_histogram * mask
    # Find secondary direction
    secondary_bin_idx = np.argmax(masked_histogram)
    secondary_direction = bin_centers[secondary_bin_idx]
    return primary_direction, secondary_direction


def calculate_tidal_periods(surface_elevation, times):
    """
    Calculate tidal period statistics and ranges between consecutive tidal cycles.

    Parameters:
    -----------
    surface_elevation : array-like
        Array of modeled water surface elevations
    times : array-like
        Array of timestamps corresponding to surface positions

    Returns:
    --------
    dict
        Dictionary containing tidal period statistics and cycle-specific data
    """

    # Find peaks (high tides) with appropriate prominence
    high_tide_indices, _ = find_peaks(surface_elevation, prominence=0.05)

    # Also find troughs (low tides)
    low_tide_indices, _ = find_peaks(-surface_elevation, prominence=0.05)

    # Sort the indices chronologically
    high_tide_indices = np.sort(high_tide_indices)
    low_tide_indices = np.sort(low_tide_indices)

    # Calculate time differences between consecutive high tides (semi-diurnal period)
    high_tide_periods = []
    low_tide_periods = []
    tidal_ranges = []
    tidal_cycles_data = []

    if len(high_tide_indices) < 2:
        # Not enough peaks to calculate periods
        return {
            "average_period_seconds": 0,
            "min_period_seconds": 0,
            "max_period_seconds": 0,
            "average_period_str": "0.00h",
            "min_period_str": "0.00h",
            "max_period_str": "0.00h",
            "tide_type": "Unknown",
            "average_range": 0,
            "min_range": 0,
            "max_range": 0,
            "min_range_cycle": None,
            "max_range_cycle": None,
            "tidal_ranges": [],
            "cycle_data": [],
        }

    # Calculate high tide periods and tidal ranges for consecutive cycles
    for i in range(1, len(high_tide_indices)):
        prev_idx = high_tide_indices[i - 1]
        curr_idx = high_tide_indices[i]

        # Find the low tide(s) between consecutive high tides
        between_low_indices = low_tide_indices[
            (low_tide_indices > prev_idx) & (low_tide_indices < curr_idx)
        ]

        # Only process valid tidal cycles with low tides between consecutive high tides
        if len(between_low_indices) > 0:
            # Use the lowest low tide between consecutive high tides
            lowest_low_idx = between_low_indices[
                np.argmin(surface_elevation[between_low_indices])
            ]

            # Calculate range for this tidal cycle
            high_tide_value = surface_elevation[prev_idx]
            low_tide_value = surface_elevation[lowest_low_idx]
            tidal_range = high_tide_value - low_tide_value
            tidal_ranges.append(tidal_range)

            # Get timestamps if available
            high_tide_time = None
            low_tide_time = None

            if times is not None:
                try:
                    if isinstance(times, pd.DatetimeIndex):
                        high_tide_time = times[prev_idx]
                        low_tide_time = times[lowest_low_idx]
                    elif hasattr(times, "iloc"):
                        high_tide_time = times.iloc[prev_idx]
                        low_tide_time = times.iloc[lowest_low_idx]
                    else:
                        high_tide_time = times[prev_idx]
                        low_tide_time = times[lowest_low_idx]
                except Exception:
                    pass

            # Record detailed data for this tidal cycle
            cycle_data = {
                "high_tide_index": prev_idx,
                "high_tide_value": high_tide_value,
                "high_tide_time": high_tide_time,
                "low_tide_index": lowest_low_idx,
                "low_tide_value": low_tide_value,
                "low_tide_time": low_tide_time,
                "tidal_range": tidal_range,
            }
            tidal_cycles_data.append(cycle_data)

        # Calculate time difference between consecutive high tides
        if times is not None:
            try:
                if isinstance(times, pd.DatetimeIndex):
                    time_diff = (times[curr_idx] - times[prev_idx]).total_seconds()
                elif hasattr(times, "iloc"):
                    time_diff = (
                        times.iloc[curr_idx] - times.iloc[prev_idx]
                    ).total_seconds()
                else:
                    time_diff = (times[curr_idx] - times[prev_idx]).total_seconds()

                # Only include reasonable periods (10-14 hours for semi-diurnal, 20-26 hours for diurnal)
                if (
                    10 * 3600 < time_diff < 14 * 3600
                    or 20 * 3600 < time_diff < 26 * 3600
                ):
                    high_tide_periods.append(time_diff)
            except Exception:
                continue

    # Calculate statistics for tidal periods
    all_periods = high_tide_periods + low_tide_periods

    # Calculate statistics if we have valid periods
    if all_periods:
        avg_period = np.mean(all_periods)
        min_period = np.min(all_periods)
        max_period = np.max(all_periods)

        # Use plain language descriptions for tide patterns
        if 10 * 3600 < avg_period < 14 * 3600:
            tide_type = "Twice Daily Tides"
        elif 20 * 3600 < avg_period < 26 * 3600:
            tide_type = "Once Daily Tides"
        else:
            tide_type = "Mixed Pattern Tides"
    else:
        # No valid periods found
        return {
            "average_period_seconds": 0,
            "min_period_seconds": 0,
            "max_period_seconds": 0,
            "average_period_str": "0.00h",
            "min_period_str": "0.00h",
            "max_period_str": "0.00h",
            "tide_type": "Unknown",
            "average_range": 0,
            "min_range": 0,
            "max_range": 0,
            "min_range_cycle": None,
            "max_range_cycle": None,
            "tidal_ranges": [],
            "cycle_data": tidal_cycles_data,
        }

    # Calculate tidal range statistics
    if tidal_ranges:
        avg_range = np.mean(tidal_ranges)
        min_range = np.min(tidal_ranges)
        max_range = np.max(tidal_ranges)

        # Find the indices of min and max ranges for reporting
        min_range_idx = np.argmin(tidal_ranges)
        max_range_idx = np.argmax(tidal_ranges)

        # Get the corresponding cycle data
        min_range_cycle = (
            tidal_cycles_data[min_range_idx]
            if min_range_idx < len(tidal_cycles_data)
            else None
        )
        max_range_cycle = (
            tidal_cycles_data[max_range_idx]
            if max_range_idx < len(tidal_cycles_data)
            else None
        )
    else:
        avg_range = 0
        min_range = 0
        max_range = 0
        min_range_cycle = None
        max_range_cycle = None

    def format_seconds_to_decimal_hours(seconds):
        # Convert seconds to hours (as a float)
        hours = seconds / 3600

        # Format to 2 decimal places and add 'h' suffix
        return f"{hours:.2f}h"

    # Create the return dictionary with all needed keys
    period_stats = {
        "average_period_seconds": avg_period if all_periods else 0,
        "min_period_seconds": min_period if all_periods else 0,
        "max_period_seconds": max_period if all_periods else 0,
        "average_period_str": format_seconds_to_decimal_hours(avg_period)
        if all_periods
        else "0.00h",
        "min_period_str": format_seconds_to_decimal_hours(min_period)
        if all_periods
        else "0.00h",
        "max_period_str": format_seconds_to_decimal_hours(max_period)
        if all_periods
        else "0.00h",
        "tide_type": tide_type if all_periods else "Unknown",
        "average_range": avg_range,
        "min_range": min_range,
        "max_range": max_range,
        "min_range_cycle": min_range_cycle,
        "max_range_cycle": max_range_cycle,
        "tidal_ranges": tidal_ranges,
        "cycle_data": tidal_cycles_data,
    }

    return period_stats


def calculate_tidal_levels(surface_positions, msl_tolerance_meters=0.2):
    """
    Calculate model-derived tidal reference levels using plain language terminology.
    Converts from NAVD88 datum to Mean Sea Level (MSL) relative values.

    Parameters:
    -----------
    surface_positions : array-like
        Array of modeled water surface elevations (referenced to NAVD88)
    msl_tolerance_meters : float, optional
        Tolerance for validating MSL conversion in meters (default: 0.2)
        Used to check if mean water level is close to zero after conversion

    Returns:
    --------
    dict
        Dictionary containing tidal reference levels relative to MSL with plain language keys

    Raises:
    -------
    ValueError
        If mean water level is more than msl_tolerance_meters from zero after MSL conversion,
        indicating the conversion may not be working properly

    Notes:
    ------
    This function converts NAVD88-referenced surface elevations to MSL-relative tidal levels:
    1. Calculates MSL offset (mean of all surface positions)
    2. Converts all tidal levels to be relative to MSL (subtracts offset)
    3. Validates that resulting mean water level is close to zero
    """
    import numpy as np
    from scipy.signal import find_peaks

    # Convert to numpy array for calculations
    surface_positions = np.array(surface_positions)

    # Calculate MSL offset - this is the mean water level relative to NAVD88
    msl_offset_from_navd88 = np.mean(surface_positions)

    # Convert surface positions to be relative to MSL
    # Subtract the offset so that mean becomes ~0
    surface_relative_to_msl = surface_positions - msl_offset_from_navd88

    # Validate the conversion worked (mean should now be very close to 0)
    converted_mean = np.mean(surface_relative_to_msl)
    if abs(converted_mean) > msl_tolerance_meters:
        raise ValueError(
            f"Error: Converted MSL ({converted_mean:.2e}) is greater than {msl_tolerance_meters} m from zero."
        )

    print(f"MSL offset from NAVD88: {msl_offset_from_navd88:.3f} m")
    print(f"Converted mean relative to MSL: {converted_mean:.6f} m")

    # Find peaks (high tides) and troughs (low tides) using MSL-relative data
    tidal_range_calculation_method = "peak_detection"
    high_tide_indices, _ = find_peaks(surface_relative_to_msl, prominence=0.05)
    low_tide_indices, _ = find_peaks(-surface_relative_to_msl, prominence=0.05)

    # Sort the indices chronologically
    # If no peaks or troughs are found, use fallback method
    if len(high_tide_indices) == 0 or len(low_tide_indices) == 0:
        tidal_range_calculation_method = "top_20%"
        print("Warning: Could not detect peaks and troughs. Using simplified method.")
        high_tides = np.sort(surface_relative_to_msl)[
            -int(len(surface_relative_to_msl) * 0.2) :
        ]  # Top 20%
        low_tides = np.sort(surface_relative_to_msl)[
            : int(len(surface_relative_to_msl) * 0.2)
        ]  # Bottom 20%

        # Create simple indices for reference
        high_tide_indices = np.argsort(surface_relative_to_msl)[
            -int(len(surface_relative_to_msl) * 0.2) :
        ]
        low_tide_indices = np.argsort(surface_relative_to_msl)[
            : int(len(surface_relative_to_msl) * 0.2)
        ]
    else:
        # Get the water levels at high and low tides (MSL-relative)
        high_tides = surface_relative_to_msl[high_tide_indices]
        low_tides = surface_relative_to_msl[low_tide_indices]

    # Calculate tidal statistics relative to MSL
    max_high_tide = np.max(high_tides)  # Highest high tide above MSL
    min_high_tide = np.min(high_tides)  # Lowest high tide above MSL
    mean_high_tide = np.mean(high_tides)  # Mean high tide above MSL
    mean_water_level = np.mean(surface_relative_to_msl)  # Should be ~0
    max_low_tide = np.max(low_tides)  # Highest low tide (could be above MSL)
    mean_low_tide = np.mean(low_tides)  # Mean low tide below MSL
    min_low_tide = np.min(low_tides)  # Lowest low tide below MSL

    # Calculate tidal range
    tidal_range = max_high_tide - min_low_tide

    # Validation checks for physically reasonable tidal levels
    validation_warnings = []

    # Check if tidal range is reasonable (typically 0.1m to 15m globally)
    if tidal_range < 0.1:
        validation_warnings.append(f"Very small tidal range ({tidal_range:.3f} m)")
    elif tidal_range > 15.0:
        validation_warnings.append(f"Very large tidal range ({tidal_range:.3f} m)")

    # Check if high tides are actually higher than low tides
    if mean_high_tide <= mean_low_tide:
        validation_warnings.append("Mean high tide is not greater than mean low tide")

    # Check if the mean is close to zero after conversion
    if abs(mean_water_level) > msl_tolerance_meters:
        raise ValueError(
            f"MSL conversion validation failed: Mean water level ({mean_water_level:.3f} m) "
            f"is more than {msl_tolerance_meters} m from zero after conversion to MSL. "
            f"This indicates the conversion may not be working properly. "
            f"Expected range: [{-msl_tolerance_meters:.1f}, {msl_tolerance_meters:.1f}] m relative to MSL."
        )

    # Print validation results
    if validation_warnings:
        print("Validation warnings:")
        for warning in validation_warnings:
            print(f"  - {warning}")
    else:
        print("All tidal level validations passed")

    # Create dictionary with MSL-relative tidal levels
    tidal_data = {
        "Tidal Range Calculation Method": tidal_range_calculation_method,
        # Tidal levels relative to MSL
        "Max High Tide": max_high_tide,  # Maximum high tide above MSL
        "Min High Tide": min_high_tide,  # Minimum high tide above MSL
        "Mean High Tide": mean_high_tide,  # Average high tide above MSL
        "Mean Water Level": mean_water_level,  # Should be ~0 (MSL reference)
        "Max Low Tide": max_low_tide,  # Maximum low tide relative to MSL
        "Mean Low Tide": mean_low_tide,  # Average low tide relative to MSL
        "Min Low Tide": min_low_tide,  # Minimum low tide below MSL
        # Conversion metadata
        "MSL_Offset_from_NAVD88": msl_offset_from_navd88,  # Offset applied for conversion
        "Tidal_Range": tidal_range,  # Max high - min low tide
        # Indices for reference
        "high_tide_indices": high_tide_indices,
        "low_tide_indices": low_tide_indices,
        # Validation metadata
        "msl_conversion_successful": len(validation_warnings) == 0,
        "validation_warnings": validation_warnings,
        "msl_tolerance_used": msl_tolerance_meters,
    }

    return tidal_data


class VAPSummaryCalculator:
    """
    Class for calculating averages, max values, and 95th percentiles across VAP NetCDF files.
    Handles both yearly and monthly processing with a unified approach.
    """

    def __init__(
        self, config, location_name, face_batch_size=None, batch_index_start=0
    ):
        self.config = config
        self.location_name = location_name
        self.location = config["location_specification"][location_name]
        self.constant_variables = ["nv"]  # Variables that should remain constant

        # Face batching parameters
        self.face_batch_size = face_batch_size
        self.batch_index_start = batch_index_start
        self.batch_number = batch_index_start

        # Common paths
        self.vap_path = file_manager.get_vap_output_dir(config, self.location)
        self.vap_nc_files = sorted(list(self.vap_path.rglob("*.nc")))

        # Verify minimum files
        if len(self.vap_nc_files) < 12:
            raise ValueError(
                f"Expecting at least 12 files in {self.vap_path}, found {len(self.vap_nc_files)}"
            )

        # Calculate expected timestamps
        self.days_per_year = 365
        if self.location["output_name"] == "WA_puget_sound":
            # Puget sound is missing one day
            self.days_per_year = 364

        self.seconds_per_year = self.days_per_year * 24 * 60 * 60
        self.expected_timestamps = int(
            self.seconds_per_year / self.location["expected_delta_t_seconds"]
        )

        # Track max values across all files for each variable
        self.max_values = {}

        # Identify variable types
        self.avg_vars = []  # Regular variables for averaging
        self.max_vars = []  # Max variables
        self.p95_vars = []  # 95th percentile variables

    def _calculate_face_batch_slice(self, total_faces):
        """
        Calculate the face slice for the current batch configuration.
        Args:
            total_faces: Total number of faces in the dataset
        Returns:
            slice: Python slice object for face dimension, or None for full dataset
        """
        if self.face_batch_size is None:
            return None

        start_index = self.batch_number * self.face_batch_size
        end_index = min(start_index + self.face_batch_size, total_faces)

        if start_index >= total_faces:
            raise ValueError(
                f"Batch {self.batch_number} starts at face {start_index}, but only {total_faces} faces available"
            )

        return slice(start_index, end_index)

    def _load_dataset_with_face_batch(self, nc_file):
        """
        Load a dataset with optional face batching applied.

        Args:
            nc_file: Path to NetCDF file

        Returns:
            xarray.Dataset: Loaded dataset with face batching applied if configured
        """
        ds = nc_manager.nc_open(nc_file, self.config)

        # Apply face batching if configured
        if self.face_batch_size is not None:
            total_faces = ds.sizes["face"]
            face_slice = self._calculate_face_batch_slice(total_faces)

            if face_slice is not None:
                ds = ds.isel(face=face_slice)

        return ds

    def _verify_timestamps(self):
        print("Verifying timestamps across all files...")
        verify_timestamps(
            self.vap_nc_files,
            self.expected_timestamps,
            self.location["expected_delta_t_seconds"],
            self.config,
        )

    def _verify_constant_vars(self, ds, first_ds):
        """Verify constant variables haven't changed."""
        for var in self.constant_variables:
            if var in list(ds.keys()) and var in list(first_ds.keys()):
                if not np.array_equal(ds[var].values, first_ds[var].values):
                    print(
                        f"WARNING: Variable '{var}' differs between files. Using value from first file."
                    )

    def _initialize_dataset(self, dataset, constant_variables):
        """
        Initialize a template dataset and identify variable types.
        """
        # Initialize with the first timestep
        template = dataset.isel(time=0).copy(deep=True)
        first_timestamp = dataset.time.isel(time=0).values
        time_attrs = dataset.time.attrs.copy()

        # Store all variable attributes
        var_attrs = {}
        for var in dataset.data_vars:
            var_attrs[var] = dataset[var].attrs.copy()

            # Identify variables by type
            if "time" in dataset[var].dims and var not in constant_variables:
                if "water_column_max" in var:
                    self.max_vars.append(var)
                elif "water_column_95th_percentile" in var:
                    self.p95_vars.append(var)
                else:
                    self.avg_vars.append(var)

        # Initialize variables with appropriate data types
        for var in template.data_vars:
            if "time" in dataset[var].dims and var not in constant_variables:
                # Use float64 for better numerical stability
                if np.issubdtype(template[var].dtype, np.integer):
                    template[var] = template[var].astype(np.float64)
                elif np.issubdtype(template[var].dtype, np.floating):
                    template[var] = template[var].astype(np.float64)

                # Initialize to zero (will be properly set during processing)
                template[var].values.fill(0)

                # Initialize max_values tracking for max variables
                if var in self.max_vars:
                    self.max_values[var] = None

        return template, first_timestamp, time_attrs, var_attrs

    def _update_averages(self, result_ds, dataset, count):
        """
        Update running averages for regular variables only.
        """
        current_times = len(dataset.time)

        # Update running average for regular variables only
        for var in self.avg_vars:
            if var in dataset.data_vars:
                # Calculate mean for this dataset
                if np.issubdtype(dataset[var].dtype, np.integer):
                    # Safely convert integers to float64 first
                    file_avg = dataset[var].astype(np.float64).mean(dim="time")
                else:
                    file_avg = dataset[var].mean(dim="time")

                # Update the rolling average using the formula:
                # new_avg = old_avg + (new_value - old_avg) / new_count
                if count == 0:
                    # First iteration - simply set to the file average
                    result_ds[var] = file_avg
                else:
                    weight = current_times / (count + current_times)
                    result_ds[var] = (
                        result_ds[var] + (file_avg - result_ds[var]) * weight
                    )

        return count + current_times

    def _update_max_values(self, result_ds, dataset, count):
        """
        Update max values and track them for 95th percentile calculation.
        """
        # Process each max variable
        for var in self.max_vars:
            if var in dataset.data_vars:
                # Calculate the max for this dataset
                file_max = dataset[var].max(dim="time")

                # Track all max values for later 95th percentile calculation
                if self.max_values[var] is None:
                    self.max_values[var] = file_max.expand_dims(dim={"file": [count]})
                else:
                    # Add to our collection of max values
                    new_max = file_max.expand_dims(dim={"file": [count]})
                    self.max_values[var] = xr.concat(
                        [self.max_values[var], new_max], dim="file"
                    )

                # For the running result, take the maximum of the current max and the previous max
                if count == 0:
                    # First iteration - simply set to the file max
                    result_ds[var] = file_max
                else:
                    # Update the max using element-wise maximum
                    result_ds[var] = xr.where(
                        file_max > result_ds[var], file_max, result_ds[var]
                    )

        return result_ds

    def _calculate_percentiles(self, result_ds):
        """
        Calculate 95th percentiles of the max values after all files have been processed.
        """
        print("Calculating 95th percentiles of max values...")

        # Map each p95 variable to its corresponding max variable
        max_to_p95_map = {}
        for p95_var in self.p95_vars:
            # Find the corresponding max variable by replacing '95th_percentile' with 'max'
            base_name = p95_var.replace("_95th_percentile_", "_max_")
            if base_name in self.max_vars:
                max_to_p95_map[base_name] = p95_var

        # Calculate 95th percentile for each p95 variable
        for max_var, p95_var in max_to_p95_map.items():
            if max_var in self.max_values and self.max_values[max_var] is not None:
                # Calculate the 95th percentile along the file dimension
                p95_value = self.max_values[max_var].quantile(0.95, dim="file")
                result_ds[p95_var] = p95_value
                print(f"Calculated 95th percentile for {p95_var} from {max_var}")

        return result_ds

    def calculate_to_direction_qoi(self, result_ds, to_direction_data):
        """
        Calculate direction quantities of interest (QOI) using accumulated direction data.

        Args:
            result_ds: Result dataset to add direction QOI variables to
            to_direction_data: Accumulated direction data across all processed files

        Returns:
            xarray.Dataset: Updated dataset with direction QOI variables
        """
        if to_direction_data is None or to_direction_data.size == 0:
            print("Warning: No direction data available for QOI calculation")
            return result_ds

        # Direction data shape is [time, sigma_layer, face]
        # We'll calculate QOI for each face and sigma_layer combination
        n_sigma_layers, n_faces = to_direction_data.shape[1], to_direction_data.shape[2]

        # Initialize arrays for primary and secondary directions
        primary_directions = np.full((n_sigma_layers, n_faces), np.nan)
        secondary_directions = np.full((n_sigma_layers, n_faces), np.nan)
        bin_width_degrees = 2
        mask_width_degrees = 180

        # Calculate direction QOI for each sigma_layer-face combination
        for layer_idx in range(n_sigma_layers):
            for face_idx in range(n_faces):
                # Extract time series for this sigma_layer-face combination
                direction_timeseries = to_direction_data[:, layer_idx, face_idx]

                # Remove NaN values
                valid_mask = ~np.isnan(direction_timeseries)
                if np.sum(valid_mask) < 10:  # Need minimum data points
                    continue

                valid_directions = direction_timeseries[valid_mask]

                # Calculate primary and secondary directions
                primary_dir, secondary_dir = (
                    calculate_single_primary_and_secondary_direction(
                        valid_directions,
                        bin_width_degrees=bin_width_degrees,
                        mask_width_degrees=mask_width_degrees,
                    )
                )
                primary_directions[layer_idx, face_idx] = primary_dir
                secondary_directions[layer_idx, face_idx] = secondary_dir

        # Add direction QOI variables to result dataset using correct dimension names
        # Create new variables for direction QOI with CF-compliant attributes
        result_ds["vap_sea_water_primary_to_direction"] = xr.DataArray(
            primary_directions,
            dims=["sigma_layer", "face"],
            attrs={
                "long_name": "Sea Water Primary To Direction",
                "units": "degrees",
                "valid_range": [0.0, 360.0],
                "description": "Most frequent flow direction at each location and depth based on directional histogram analysis",
                "computation": f"calculated using directional histogram with {bin_width_degrees}-degree wide bins with {mask_width_degrees}-degree masking",
                "input_variables": "vap_sea_water_to_direction",
                "cell_methods": "time: histogram_mode",
            },
        )

        result_ds["vap_sea_water_secondary_to_direction"] = xr.DataArray(
            secondary_directions,
            dims=["sigma_layer", "face"],
            attrs={
                "long_name": "sea water secondary flow direction",
                "standard_name": "sea_water_to_direction",
                "units": "degrees",
                "valid_range": [0.0, 360.0],
                "description": "Second most frequent flow direction at each location and depth, excluding directions within 90 degrees of primary direction",
                "computation": f"calculated using directional histogram with {bin_width_degrees}-degree wide bins with {mask_width_degrees}-degree masking",
                "input_variables": "vap_sea_water_to_direction",
                "cell_methods": "time: histogram_mode",
            },
        )

        return result_ds

    def calculate_surface_elevation_qoi(
        self, result_ds, zeta_center_data, all_timestamps
    ):
        """
        Calculate surface elevation quantities of interest (QOI) using accumulated surface elevation data.

        Args:
            result_ds: Result dataset to add surface elevation QOI variables to
            zeta_center_data: Accumulated surface elevation data across all processed files
            all_timestamps: All timestamps corresponding to the surface elevation data

        Returns:
            xarray.Dataset: Updated dataset with surface elevation QOI variables
        """

        # Assuming zeta_center_data shape is [time, face] for surface elevation
        n_faces = (
            zeta_center_data.shape[1]
            if len(zeta_center_data.shape) > 1
            else zeta_center_data.shape[0]
        )

        # Initialize arrays for tidal statistics
        mean_water_levels = np.full(n_faces, np.nan)
        max_high_tides = np.full(n_faces, np.nan)
        mean_high_tides = np.full(n_faces, np.nan)
        mean_low_tides = np.full(n_faces, np.nan)
        min_low_tides = np.full(n_faces, np.nan)
        tidal_ranges = np.full(n_faces, np.nan)

        # Tidal period statistics
        avg_periods = np.full(n_faces, np.nan)
        min_periods = np.full(n_faces, np.nan)
        max_periods = np.full(n_faces, np.nan)

        # Calculate tidal statistics for each face
        for face_idx in range(n_faces):
            # Extract time series for this face
            surface_timeseries = zeta_center_data[:, face_idx]

            # Get corresponding valid timestamps if available

            # Calculate tidal levels using the provided function
            tidal_levels = calculate_tidal_levels(surface_timeseries)
            print(
                f"Face {face_idx}: MSL: {tidal_levels['Mean Water Level']:.3f} m, Max High Tide: {tidal_levels['Max High Tide']:.3f} m, Min Low Tide: {tidal_levels['Min Low Tide']:.3f} m"
            )

            # Calculate tidal periods if timestamps are available
            period_stats = calculate_tidal_periods(surface_timeseries, all_timestamps)
            print(
                f"Face {face_idx}: Tidal Periods: Max: {period_stats['max_period_seconds']:.2f}s, Min : {period_stats['min_period_seconds']:.2f}s, Avg: {period_stats['average_period_seconds']:.2f}s"
            )

            avg_periods[face_idx] = period_stats["average_period_seconds"]
            min_periods[face_idx] = period_stats["min_period_seconds"]
            max_periods[face_idx] = period_stats["max_period_seconds"]

            # Store tidal level results
            mean_water_levels[face_idx] = tidal_levels["Mean Water Level"]
            max_high_tides[face_idx] = tidal_levels["Max High Tide"]
            mean_high_tides[face_idx] = tidal_levels["Mean High Tide"]
            mean_low_tides[face_idx] = tidal_levels["Mean Low Tide"]
            min_low_tides[face_idx] = tidal_levels["Min Low Tide"]
            tidal_ranges[face_idx] = (
                tidal_levels["Max High Tide"] - tidal_levels["Min Low Tide"]
            )

        # Add surface elevation QOI variables to result dataset with CF-compliant attributes
        result_ds["vap_sea_surface_elevation_mean"] = xr.DataArray(
            mean_water_levels,
            dims=["face"],
            attrs={
                "long_name": "Mean Sea Surface Elevation",
                "units": "m",
                "description": "Average water surface elevation over the analysis period",
                "computation": "arithmetic mean of all surface elevation values",
                "input_variables": "vap_zeta_center",
                "cell_methods": "time: mean",
            },
        )

        result_ds["vap_sea_surface_elevation_high_tide_max"] = xr.DataArray(
            max_high_tides,
            dims=["face"],
            attrs={
                "long_name": "maximum high tide sea surface height above mean sea surface elevation",
                "units": "m",
                "description": "Highest observed high tide level detected using peak analysis",
                "computation": "maximum value among detected high tide peaks",
                "input_variables": "vap_zeta_center",
                "cell_methods": "time: maximum within high_tide_events",
            },
        )

        result_ds["vap_surface_elevation_high_tide_mean"] = xr.DataArray(
            mean_high_tides,
            dims=["face"],
            attrs={
                "long_name": "mean high tide sea surface height above mean sea surface elevation",
                "units": "m",
                "description": "Average of all high tide levels detected using peak analysis",
                "computation": "arithmetic mean of detected high tide peak values",
                "input_variables": "vap_zeta_center",
                "cell_methods": "time: mean within high_tide_events",
            },
        )

        result_ds["vap_surface_elevation_low_tide_mean"] = xr.DataArray(
            mean_low_tides,
            dims=["face"],
            attrs={
                "long_name": "mean low tide sea surface height above mean sea surface elevation",
                "units": "m",
                "description": "Average of all low tide levels detected using trough analysis",
                "computation": "arithmetic mean of detected low tide trough values",
                "input_variables": "vap_zeta_center",
                "cell_methods": "time: mean within low_tide_events",
            },
        )

        result_ds["vap_surface_elevation_low_tide_min"] = xr.DataArray(
            min_low_tides,
            dims=["face"],
            attrs={
                "long_name": "minimum low tide sea surface height above mean sea surface elevation",
                "units": "m",
                "description": "Lowest observed low tide level detected using trough analysis",
                "computation": "minimum value among detected low tide troughs",
                "input_variables": "vap_zeta_center",
                "cell_methods": "time: minimum within low_tide_events",
            },
        )

        result_ds["vap_tidal_range"] = xr.DataArray(
            tidal_ranges,
            dims=["face"],
            attrs={
                "long_name": "tidal range",
                "units": "m",
                "description": "Difference between maximum high tide and minimum low tide levels",
                "computation": "vap_sea_surface_elevation_high_tide_max - vap_surface_elevation_low_tide_min",
                "input_variables": "vap_zeta_center",
                "cell_methods": "time: range",
            },
        )

        result_ds["vap_average_tidal_period"] = xr.DataArray(
            avg_periods,
            dims=["face"],
            attrs={
                "long_name": "average tidal period",
                "units": "s",
                "description": "Average time between consecutive high tides or low tides",
                "computation": "arithmetic mean of time differences between consecutive tidal peaks",
                "input_variables": "vap_zeta_center",
                "cell_methods": "time: mean of tidal_periods",
            },
        )

        result_ds["vap_min_tidal_period"] = xr.DataArray(
            min_periods,
            dims=["face"],
            attrs={
                "long_name": "minimum tidal period",
                "units": "s",
                "description": "Shortest time between consecutive high tides or low tides",
                "computation": "minimum of time differences between consecutive tidal peaks",
                "input_variables": "vap_zeta_center",
                "cell_methods": "time: minimum of tidal_periods",
            },
        )

        result_ds["vap_max_tidal_period"] = xr.DataArray(
            max_periods,
            dims=["face"],
            attrs={
                "long_name": "maximum tidal period",
                "units": "s",
                "description": "Longest time between consecutive high tides or low tides",
                "computation": "maximum of time differences between consecutive tidal peaks",
                "input_variables": "vap_zeta_center",
                "cell_methods": "time: maximum of tidal_periods",
            },
        )

        return result_ds

    def _process_files(self, ds_template, file_list, time_filter=None):
        """
        Process a list of files, updating averages, max values, and 95th percentiles.

        Args:
            ds_template: Template dataset initialized with proper dimensions
            file_list: List of files to process
            time_filter: Optional function to filter timestamps in each dataset

        Returns:
            tuple: (updated_ds, first_timestamp, time_attrs, var_attrs, source_files)
        """
        result_ds = ds_template
        count = 0
        source_files = []
        first_timestamp = None
        time_attrs = None
        var_attrs = {}
        first_ds = None

        # Initialize lists to accumulate direction and surface elevation data with timestamps
        to_direction_data = []
        zeta_center_data = []
        all_timestamps = []

        # Process each file
        for i, nc_file in enumerate(file_list):
            print(f"Processing File {i}: {nc_file}")

            # Load dataset with face batching applied
            ds = self._load_dataset_with_face_batch(nc_file)

            # Apply time filter if provided
            if time_filter is not None:
                ds_filtered = time_filter(ds)
                if ds_filtered is None or len(ds_filtered.time) == 0:
                    ds.close()
                    continue  # Skip if no data after filtering
                ds = ds_filtered

            # Initialize if this is the first valid file
            if first_ds is None:
                result_ds, first_timestamp, time_attrs, var_attrs = (
                    self._initialize_dataset(ds, self.constant_variables)
                )
                first_ds = ds.copy()
            else:
                self._verify_constant_vars(ds, first_ds)

            # Update averages and max values separately
            count = self._update_averages(result_ds, ds, count)
            result_ds = self._update_max_values(result_ds, ds, count)

            # Accumulate direction and surface elevation data with timestamps
            to_direction_data.append(ds["vap_sea_water_to_direction"].values)

            zeta_center_data.append(ds["vap_zeta_center"].values)

            all_timestamps.append(ds.time.values)

            # Track source files
            if str(nc_file) not in source_files:
                source_files.append(str(nc_file))

            ds.close()

        # Calculate 95th percentiles after processing all files
        if first_ds is not None:  # Only if we processed at least one file
            result_ds = self._calculate_percentiles(result_ds)

        # Combine accumulated data and calculate QOI
        # Concatenate along time axis (axis 0)
        combined_to_direction = np.concatenate(to_direction_data, axis=0)
        result_ds = self.calculate_to_direction_qoi(result_ds, combined_to_direction)

        # Concatenate along time axis (axis 0)
        combined_zeta_center = np.concatenate(zeta_center_data, axis=0)

        # Combine timestamps
        combined_timestamps = np.concatenate(all_timestamps, axis=0)

        result_ds = self.calculate_surface_elevation_qoi(
            result_ds, combined_zeta_center, combined_timestamps
        )

        return result_ds, first_timestamp, time_attrs, var_attrs, source_files

    def _finalize_dataset(self, running_avg, first_timestamp, time_attrs, var_attrs):
        """
        Finalize the averaged dataset by restoring attributes and metadata.

        Args:
            running_avg: Running average dataset
            first_timestamp: First timestamp for time coordinate
            time_attrs: Time dimension attributes
            var_attrs: Variable attributes dictionary

        Returns:
            xarray.Dataset: Finalized dataset with restored attributes
        """
        # Restore variable attributes and potentially convert back to original types
        for var in running_avg.data_vars:
            if (
                var not in running_avg.dims
                and var not in running_avg.coords
                and var not in self.constant_variables
            ):
                # Restore variable attributes
                if var in var_attrs:
                    running_avg[var].attrs = var_attrs[var]

                    # Optionally convert back to original type if safe and configured
                    original_dtype = var_attrs[var].get("_original_dtype")
                    if original_dtype and self.config.get("preserve_dtypes", False):
                        try:
                            # Test conversion for safety
                            test_conversion = running_avg[var].values.astype(
                                original_dtype
                            )
                            running_avg[var] = running_avg[var].astype(original_dtype)
                        except (OverflowError, ValueError):
                            print(
                                f"WARNING: Cannot safely convert {var} back to {original_dtype}. Keeping higher precision."
                            )

        # Restore the time coordinate with its attributes
        running_avg = running_avg.assign_coords(time=("time", [first_timestamp]))
        running_avg["time"].attrs = time_attrs  # Restore time attributes

        return running_avg

    def _save_dataset(
        self,
        dataset,
        output_path,
        filename,
        source_files,
        data_level,
        temporal="average",
    ):
        """
        Save the dataset to a NetCDF file with proper attributes and encoding.
        Modified to include face batch information in filename if applicable.

        Args:
            dataset: xarray Dataset to save
            output_path: Directory path to save to
            filename: Base filename
            source_files: List of source files used to create this dataset
            data_level: Data level identifier
            temporal: Temporal specification for filename

        Returns:
            Path: Path to the saved file
        """
        # Generate base filename
        data_level_file_name = (
            file_name_convention_manager.generate_filename_for_data_level(
                dataset,
                self.location["output_name"],
                self.config["dataset"]["name"],
                data_level,
                temporal=temporal,
            )
        )

        # Add face batch information to filename if batching is used
        if self.face_batch_size is not None:
            end_face = (
                self.batch_index_start
                + min(
                    self.face_batch_size,
                    dataset.dims.get("face", self.face_batch_size),
                )
                - 1
            )
            batch_info = f"_faces_{self.batch_index_start}_{end_face}"
            # Insert before file extension
            parts = data_level_file_name.rsplit(".", 1)
            if len(parts) == 2:
                data_level_file_name = f"{parts[0]}{batch_info}.{parts[1]}"
            else:
                data_level_file_name = f"{data_level_file_name}{batch_info}"

        # Add standard attributes
        dataset = attrs_manager.standardize_dataset_global_attrs(
            dataset,
            self.config,
            self.location,
            data_level,
            source_files,
        )

        # Add face batch metadata
        if self.face_batch_size is not None:
            dataset.attrs["face_batch_size"] = self.face_batch_size
            dataset.attrs["batch_index_start"] = self.batch_index_start
            dataset.attrs["face_dimension"] = "face"

        file_path = Path(output_path, filename.format(data_level_file_name))
        print(f"Saving to {file_path}...")

        nc_manager.nc_write(dataset, file_path, self.config)

        return file_path

    def calculate_yearly_average(self):
        """
        Calculate yearly average values, max values, and 95th percentiles across VAP NC files.
        """
        output_path = file_manager.get_yearly_summary_vap_output_dir(
            self.config, self.location
        )

        # Temporarily disable existing file check
        # output_nc_files = list(output_path.rglob("*.nc"))

        # if len(output_nc_files) > 0:
        #     print(
        #         f"{len(output_nc_files)} summary files already exist. Skipping yearly averaging."
        #     )
        #     return None

        # Verify timestamps
        self._verify_timestamps()

        # Reset variable tracking for this calculation
        self.avg_vars = []
        self.max_vars = []
        self.p95_vars = []
        self.max_values = {}

        # Process all files
        print(f"Starting yearly processing of {len(self.vap_nc_files)} vap files...")
        result_ds, first_timestamp, time_attrs, var_attrs, source_files = (
            self._process_files(None, self.vap_nc_files)
        )

        if result_ds is None:
            print("No data found for yearly processing")
            return None

        print("Finalizing yearly dataset...")
        # Finalize the dataset
        finalized_ds = self._finalize_dataset(
            result_ds, first_timestamp, time_attrs, var_attrs
        )

        print(finalized_ds.info())
        print(finalized_ds.time)

        # Save the yearly average
        return self._save_dataset(
            finalized_ds,
            output_path,
            "001.{}",
            source_files,
            "b3",
            temporal="1_year_average",
        )

    def calculate_monthly_averages(self):
        """
        Calculate monthly average values, max values, and 95th percentiles across VAP NC files.
        """
        output_path = file_manager.get_monthly_summary_vap_output_dir(
            self.config, self.location
        )
        # Create the output directory if it doesn't exist
        output_path.mkdir(parents=True, exist_ok=True)

        # Check if monthly files already exist
        output_nc_files = list(output_path.rglob("*.nc"))
        if len(output_nc_files) >= 12:
            print(
                f"{len(output_nc_files)} monthly summary files already exist. Skipping monthly averaging."
            )
            return output_path

        # Verify timestamps
        self._verify_timestamps()

        # Load all datasets to get the full time dimension
        print("Loading all datasets to extract timestamps...")
        all_times = []
        for nc_file in self.vap_nc_files:
            ds = nc_manager.nc_open(nc_file, self.config)
            all_times.append(ds.time.values)
            ds.close()

        # Flatten the list of arrays into a single array and convert to pandas for easier handling
        all_timestamps = pd.to_datetime(np.concatenate(all_times))

        # Group timestamps by month
        months = {}
        for timestamp in all_timestamps:
            month_key = (timestamp.year, timestamp.month)
            if month_key not in months:
                months[month_key] = []
            months[month_key].append(timestamp)

        print(f"Found data for {len(months)} distinct months")

        # Process each month separately
        for month_key, month_timestamps in months.items():
            year, month = month_key
            month_name = pd.Timestamp(year=year, month=month, day=1).strftime("%B")
            print(
                f"Processing {month_name} {year} with {len(month_timestamps)} timestamps"
            )

            # Convert back to numpy datetime64 for comparison with xarray
            month_timestamps_np = np.array(
                [np.datetime64(ts) for ts in month_timestamps]
            )

            # Define a time filter function for this month
            def month_filter(ds):
                month_mask = np.isin(ds.time.values, month_timestamps_np)
                if not any(month_mask):
                    return None
                return ds.isel(time=month_mask)

            # Reset variable tracking for this month
            self.avg_vars = []
            self.max_vars = []
            self.p95_vars = []
            self.max_values = {}

            # Process files for this month
            result_ds, first_timestamp, time_attrs, var_attrs, source_files = (
                self._process_files(None, self.vap_nc_files, time_filter=month_filter)
            )

            # Skip if no data was found for this month
            if result_ds is None:
                print(f"No data found for {month_name} {year}, skipping")
                continue

            # Finalize the dataset
            monthly_ds = self._finalize_dataset(
                result_ds, first_timestamp, time_attrs, var_attrs
            )

            # Additional monthly metadata
            monthly_ds.attrs["month"] = month
            monthly_ds.attrs["year"] = year
            monthly_ds.attrs["month_name"] = month_name

            # Save the monthly average
            month_str = f"{year}_{month:02d}"
            self._save_dataset(
                monthly_ds,
                output_path,
                f"{month:02d}.{{}}",
                source_files,
                "b2",
                temporal=f"{month_str}_average",
            )

        return output_path


def calculate_vap_yearly_average(config, location, batch_size=None, batch_number=0):
    """
    Calculate yearly averages, max values, and 95th percentiles for VAP variables.

    Args:
        config: Configuration dictionary
        location: Location name
        batch_size: Number of faces to process in this batch (None = process all)
        batch_number: Starting face index for this batch
    """
    averager = VAPSummaryCalculator(config, location, batch_size, batch_number)
    return averager.calculate_yearly_average()


def calculate_vap_monthly_average(config, location, batch_size=None, batch_number=0):
    """
    Calculate monthly averages, max values, and 95th percentiles for VAP variables.

    Args:
        config: Configuration dictionary
        location: Location name
        batch_size: Number of faces to process in this batch (None = process all)
        batch_number: Starting face index for this batch
    """
    averager = VAPSummaryCalculator(config, location, batch_size, batch_number)
    return averager.calculate_monthly_averages()


def parse_face_batch_info(filename):
    """
    Parse face batch information from filename.

    Args:
        filename: String filename containing face batch info

    Returns:
        tuple: (start_face, end_face) or (None, None) if no batch info found
    """
    # Look for pattern like "_faces_0_99" in filename
    match = re.search(r"_faces_(\d+)_(\d+)", filename)
    if match:
        start_face = int(match.group(1))
        end_face = int(match.group(2))
        return start_face, end_face
    return None, None


def get_base_filename(filename):
    """
    Get the base filename without face batch information.

    Args:
        filename: String filename

    Returns:
        str: Base filename with face batch info removed
    """
    # Remove face batch pattern from filename
    base_name = re.sub(r"_faces_\d+_\d+", "", filename)
    return base_name


def group_files_by_base(file_paths):
    """
    Group files by their base filename (without face batch info).

    Args:
        file_paths: List of Path objects

    Returns:
        dict: Dictionary mapping base filenames to lists of (path, start_face, end_face) tuples
    """
    grouped_files = defaultdict(list)

    for file_path in file_paths:
        filename = file_path.name
        start_face, end_face = parse_face_batch_info(filename)
        base_name = get_base_filename(filename)

        # Only include files with face batch information
        if start_face is not None and end_face is not None:
            grouped_files[base_name].append((file_path, start_face, end_face))
        else:
            print(f"Skipping file without face batch info: {filename}")

    # Sort each group by start_face to ensure proper ordering
    for base_name in grouped_files:
        grouped_files[base_name].sort(key=lambda x: x[1])  # Sort by start_face

    return grouped_files


def verify_face_continuity(file_info_list):
    """
    Verify that face batches are continuous and don't overlap.

    Args:
        file_info_list: List of (path, start_face, end_face) tuples, sorted by start_face

    Returns:
        bool: True if faces are continuous, False otherwise
    """
    if not file_info_list:
        return False

    expected_next = 0
    for i, (path, start_face, end_face) in enumerate(file_info_list):
        if start_face != expected_next:
            print(
                f"WARNING: Face discontinuity detected. Expected start {expected_next}, got {start_face} in {path.name}"
            )
            return False
        expected_next = end_face + 1

    return True


def combine_face_files(file_info_list, output_path):
    """
    Combine multiple face batch files into a single file.

    Args:
        file_info_list: List of (path, start_face, end_face) tuples, sorted by start_face
        output_path: Path where to save the combined file

    Returns:
        Path: Path to the created combined file
    """
    print(f"Combining {len(file_info_list)} face batch files into {output_path.name}")

    # Verify face continuity
    if not verify_face_continuity(file_info_list):
        print("WARNING: Face batches are not continuous. Proceeding anyway...")

    datasets = []

    # Load all datasets
    for path, start_face, end_face in file_info_list:
        print(f"  Loading {path.name} (faces {start_face}-{end_face})")
        ds = xr.open_dataset(path)
        datasets.append(ds)

    # Combine along face dimension
    print("  Concatenating datasets along face dimension...")
    combined_ds = xr.concat(datasets, dim="face")

    # Clean up attributes - remove face batch specific metadata
    attrs_to_remove = ["face_batch_size", "batch_index_start"]
    for attr in attrs_to_remove:
        if attr in combined_ds.attrs:
            del combined_ds.attrs[attr]

    # Add metadata about the combination
    combined_ds.attrs["combined_from_face_batches"] = True
    combined_ds.attrs["num_face_batch_files"] = len(file_info_list)
    combined_ds.attrs["total_faces"] = combined_ds.dims["face"]

    # Save the combined dataset
    print(f"  Saving combined dataset with {combined_ds.dims['face']} faces...")
    combined_ds.to_netcdf(output_path)

    # Close all datasets
    for ds in datasets:
        ds.close()
    combined_ds.close()

    return output_path


def combine_face_batch_files_in_directory(
    input_dir, output_dir=None, file_pattern="*.nc"
):
    """
    Combine face batch files in a directory into single files per base filename.

    Args:
        input_dir: Path to directory containing face batch files
        output_dir: Path to output directory (if None, uses input_dir)
        file_pattern: Glob pattern for files to process

    Returns:
        list: List of paths to created combined files
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir) if output_dir else input_path

    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)

    # Find all NetCDF files
    nc_files = list(input_path.glob(file_pattern))
    print(f"Found {len(nc_files)} NetCDF files in {input_path}")

    if not nc_files:
        print("No NetCDF files found.")
        return []

    # Group files by base filename
    grouped_files = group_files_by_base(nc_files)
    print(f"Grouped into {len(grouped_files)} base filename groups")

    created_files = []

    # Process each group
    for base_name, file_info_list in grouped_files.items():
        print(f"\nProcessing group: {base_name}")
        print(f"  Found {len(file_info_list)} face batch files")

        # Skip if only one file (no combining needed)
        if len(file_info_list) == 1:
            print(f"  Only one file found, skipping combination for {base_name}")
            continue

        # Generate output filename (remove any existing face batch info)
        output_filename = base_name
        output_file_path = output_path / output_filename

        # Skip if output file already exists
        if output_file_path.exists():
            print(f"  Output file already exists: {output_filename}")
            continue

        # Combine the files
        combined_file = combine_face_files(file_info_list, output_file_path)
        created_files.append(combined_file)
        print(f"  Successfully created: {output_filename}")

    return created_files


def combine_monthly_face_files(config, location):
    """
    Combine monthly face batch files into complete monthly files.

    Args:
        monthly_dir: Path to directory containing monthly face batch files
        output_dir: Path to output directory (if None, uses monthly_dir)

    Returns:
        list: List of paths to created combined files
    """
    print("=== Combining Monthly Face Batch Files ===")

    monthly_dir = file_manager.get_monthly_summary_vap_output_dir(config, location)
    output_dir = monthly_dir

    return combine_face_batch_files_in_directory(monthly_dir, output_dir)


def combine_yearly_face_files(config, location):
    """
    Combine yearly face batch files into complete yearly files.

    Args:
        yearly_dir: Path to directory containing yearly face batch files
        output_dir: Path to output directory (if None, uses yearly_dir)

    Returns:
        list: List of paths to created combined files
    """
    print("=== Combining Yearly Face Batch Files ===")

    yearly_dir = file_manager.get_yearly_summary_vap_output_dir(config, location)
    output_dir = yearly_dir
    return combine_face_batch_files_in_directory(yearly_dir, output_dir)
