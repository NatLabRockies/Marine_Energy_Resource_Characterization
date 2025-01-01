from datetime import datetime

import xarray as xr


def _format_datetime(ds):
    """Extract and format date and time strings from dataset."""
    time_var = ds.time.values[0]
    if isinstance(time_var, (int, float)):
        dt = datetime.fromtimestamp(time_var)
    else:
        dt = time_var.astype(datetime)

    date_str = dt.strftime("%Y%m%d")
    time_str = dt.strftime("%H%M%S")
    return date_str, time_str


def generate_filename_for_data_level(
    ds, location_id, dataset_name, data_level, qualifier=None, temporal=None, ext="nc"
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
    date_str, time_str = _format_datetime(ds)

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
