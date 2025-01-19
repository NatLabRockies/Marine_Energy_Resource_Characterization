import numpy as np
import pandas as pd


TIME_VARIABLE = "Times"


def standardize_fvcom_time(ds):
    # Convert string times to datetime64
    times_string = [ts.decode("utf-8") for ts in ds[TIME_VARIABLE].values]
    std_datetimes = pd.to_datetime(times_string, utc=True)
    return {
        "original": ds["time"].values,
        "Timestamp": std_datetimes,
        "datetime64[ns]": std_datetimes.values,
    }


def does_time_always_increase(time_array):
    this_series = pd.Series(time_array)
    return this_series.is_monotonic_increasing


def does_time_have_duplicates(time_array):
    this_series = pd.Series(time_array)
    return this_series.duplicated().any()


def calculate_time_delta_seconds(timestamps):
    deltas = pd.Series(timestamps).diff()
    delta_seconds = deltas.apply(lambda x: x.total_seconds())
    return {
        "mean_dt": delta_seconds.mean(),
        "min_dt": delta_seconds.min(),
        "max_dt": delta_seconds.max(),
    }


def does_time_match_specification(pandas_timestamps, expected_delta_t_seconds):
    # Check if time always increases
    if not does_time_always_increase(pandas_timestamps):
        raise ValueError("Time values must be strictly increasing")

    # Check for duplicate timestamps
    if does_time_have_duplicates(pandas_timestamps):
        raise ValueError("Duplicate timestamps found in the time array")

    # Calculate and validate time deltas
    delta_stats = calculate_time_delta_seconds(pandas_timestamps)

    # Allow for small numerical precision differences (e.g. 0.001 seconds)
    tolerance = 0.001

    if abs(delta_stats["mean_dt"] - expected_delta_t_seconds) > tolerance:
        raise ValueError(
            f"Mean time delta ({delta_stats['mean_dt']:.3f}s) does not match expected value ({expected_delta_t_seconds}s)"
        )

    if abs(delta_stats["min_dt"] - expected_delta_t_seconds) > tolerance:
        raise ValueError(
            f"Minimum time delta ({delta_stats['min_dt']:.3f}s) does not match expected value ({expected_delta_t_seconds}s)"
        )

    if abs(delta_stats["max_dt"] - expected_delta_t_seconds) > tolerance:
        raise ValueError(
            f"Maximum time delta ({delta_stats['max_dt']:.3f}s) does not match expected value ({expected_delta_t_seconds}s)"
        )

    return True


def generate_temporal_attrs(ds, max_allowable_delta_t_deviation_seconds=0):
    """
    Generate temporal attributes based on the time delta between timestamps in an xarray dataset.

    Parameters:
    -----------
    ds : xarray.Dataset
        Dataset containing a 'time' coordinate with datetime values
    max_allowable_delta_t_deviation_seconds : float, default=0
        Maximum allowed deviation from the expected delta t in seconds.
        Example: If set to 0.5 and expected delta_t is 3600 seconds (1 hour),
        actual deltas between 3599.5 and 3600.5 seconds would be accepted.
        Set to 0 for exact matching.

    Returns:
    --------
    dict
        Dictionary containing standard_name and long_name attributes

    Raises:
    -------
    ValueError
        - If timestamps are not uniformly spaced within tolerance
        - If delta_t_seconds is not supported
        - If timestamps precision is not supported
        - If time coordinate is missing required attributes
    """
    if "time" not in ds.coords:
        raise ValueError("Dataset must have a 'time' coordinate")

    time_coord = ds.time

    # Verify required time attributes
    required_attrs = ["precision", "time_zone"]
    missing_attrs = [attr for attr in required_attrs if attr not in time_coord.attrs]
    if missing_attrs:
        raise ValueError(
            f"Time coordinate missing required attributes: {missing_attrs}"
        )

    if time_coord.attrs["time_zone"] != "UTC":
        raise ValueError(f"Time zone must be UTC, got {time_coord.attrs['time_zone']}")

    # Verify timestamp precision
    supported_precisions = ["nanosecond", "second"]
    precision = time_coord.attrs["precision"]
    if precision not in supported_precisions:
        raise ValueError(
            f"Unsupported timestamp precision: {precision}. "
            f"Supported precisions are: {supported_precisions}"
        )

    if len(time_coord) < 2:
        raise ValueError("Need at least 2 timestamps to determine temporal resolution")

    # Calculate time differences based on precision
    if precision == "nanosecond":
        # For nanosecond precision, convert to seconds
        time_diffs = np.diff(time_coord.values) / np.timedelta64(1, "s")
    else:  # second precision
        # For second precision, differences are already in seconds
        time_diffs = np.diff(time_coord.values.astype("datetime64[s]").astype(float))

    first_diff = time_diffs[0]

    # Check if timestamps are uniformly spaced within the specified tolerance
    max_deviation = np.max(np.abs(time_diffs - first_diff))
    if max_deviation > max_allowable_delta_t_deviation_seconds:
        raise ValueError(
            f"Timestamps are not uniformly spaced within tolerance. "
            f"Maximum deviation is {max_deviation} seconds, "
            f"but maximum allowed deviation is {max_allowable_delta_t_deviation_seconds} seconds"
        )

    # Use the first time difference
    delta_t_seconds = float(first_diff)

    # Define supported temporal resolutions
    temporal_mappings = {
        3600: {
            "standard_name": "1h",
            "long_name": "hourly",
        },
        1800: {
            "standard_name": "30m",
            "long_name": "half-hourly",
        },
    }

    # Find matching resolution considering tolerance
    for supported_delta, attrs in temporal_mappings.items():
        if (
            abs(delta_t_seconds - supported_delta)
            <= max_allowable_delta_t_deviation_seconds
        ):
            return attrs

    raise ValueError(
        f"Unexpected delta_t_seconds: {delta_t_seconds}. "
        f"Supported values are: {list(temporal_mappings.keys())}. "
        f"Maximum allowed deviation is {max_allowable_delta_t_deviation_seconds} seconds"
    )
