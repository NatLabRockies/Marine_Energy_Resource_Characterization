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
    delta_seconds = deltas.seconds
    return {
        "mean_dt": delta_seconds.mean(),
        "min_dt": delta_seconds.min(),
        "max_dt": delta_seconds.max(),
    }
