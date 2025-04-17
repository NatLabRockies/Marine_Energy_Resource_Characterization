from datetime import datetime

import pandas as pd
import xarray as xr


def _format_datetime(data):
    """
    Extract and format date and time strings from either xarray Dataset with time coordinate
    or pandas DataFrame with datetime index.

    Parameters:
    -----------
    data : xr.Dataset or pd.DataFrame
        Dataset containing datetime information either as a coordinate or as an index

    Returns:
    --------
    tuple
        (date_string, time_string) in format (%Y%m%d, %H%M%S)
    """

    dt = None

    if isinstance(data, xr.Dataset):
        dt = pd.Timestamp(data.time.values[0])
    elif isinstance(data, pd.DataFrame):
        if isinstance(data.index, pd.DatetimeIndex):
            dt = pd.Timestamp(data.index[0])

    if dt is None:
        raise TypeError("Input must be either xarray Dataset or pandas DataFrame")

    return dt.strftime("%Y%m%d"), dt.strftime("%H%M%S")


def generate_filename_for_data_level(
    ds,
    location_id,
    dataset_name,
    data_level,
    qualifier=None,
    temporal=None,
    ext="nc",
    static_time=None,
):
    """Generate filename for processed data following ME naming convention.

    Args:
        ds: xarray Dataset containing the time variable for the first data point
        location_id: Mandatory string identifying the location where data were obtained
        dataset_name: Mandatory string identifying type of data being produced. Must not end in number
        data_level: Mandatory two-character descriptor indicating processing level
        qualifier: Optional string to distinguish dataset from others from same instrument.
                  Must not end in number
        temporal: Optional string indicating temporal resolution (e.g., '10hz', '30s', '5m', '1h')
        ext: File extension

    Returns:
        Formatted filename string: location_id.dataset_name[-qualifier][-temporal].data_level.date.time.ext
    """
    if static_time is None:
        date_str, time_str = _format_datetime(ds)
    else:
        date_str, time_str = static_time[0], static_time[1]

    components = [location_id, dataset_name]

    if qualifier:
        components[-1] += f"-{qualifier}"
    if temporal:
        components[-1] += f"-{temporal}"

    components.extend([data_level, date_str, time_str])

    return ".".join(components + [ext])


def generate_filename_for_viz(
    ds,
    location_id,
    dataset_name,
    data_level,
    plot_title,
    qualifier=None,
    temporal=None,
    ext="png",
):
    """Generate filename for visualization following ME naming convention.

    Args:
        ds: xarray Dataset containing the time variable for the first data point
        location_id: Mandatory string identifying the location where data were obtained
        dataset_name: Mandatory string identifying type of data being produced. Must not end in number
        data_level: Mandatory two-character descriptor indicating processing level
        plot_title: String describing the plot or visualization
        qualifier: Optional string to distinguish dataset from others from same instrument
        temporal: Optional string indicating temporal resolution (e.g., '10hz', '30s', '5m', '1h')
        ext: File extension for visualization (e.g., 'png', 'gif')

    Returns:
        Formatted filename string: location_id.dataset_name[-qualifier][-temporal].data_level.date.time.plot_title.ext
    """
    base = generate_filename_for_data_level(
        ds, location_id, dataset_name, data_level, qualifier, temporal, None
    )

    base = base.rsplit(".", 1)[0]
    return f"{base}.{plot_title}.{ext}"


def generate_filename_for_original_data(
    ds, location_id, dataset_name, original_filename, qualifier=None, temporal=None
):
    """Generate filename for raw data following ME naming convention.

    Args:
        ds: xarray Dataset containing the time variable for the first data point
        location_id: Mandatory string identifying the location where data were obtained
        dataset_name: Mandatory string identifying type of data being produced. Must not end in number
        original_filename: Original filename from the instrument
        qualifier: Optional string to distinguish dataset from others from same instrument
        temporal: Optional string indicating temporal resolution (e.g., '10hz', '30s', '5m', '1h')

    Returns:
        Formatted filename string: location_id.dataset_name[-qualifier][-temporal].00.date.time.raw.original_filename
    """
    base = generate_filename_for_data_level(
        ds, location_id, dataset_name, "00", qualifier, temporal, None
    )

    base = base.rsplit(".", 1)[0]
    return f"{base}.raw.{original_filename}"
