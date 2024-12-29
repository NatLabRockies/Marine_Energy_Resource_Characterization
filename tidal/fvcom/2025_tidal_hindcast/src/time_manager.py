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
