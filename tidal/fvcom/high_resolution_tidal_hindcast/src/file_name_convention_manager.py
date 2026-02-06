from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import xarray as xr


@dataclass
class DataFileName:
    """Bidirectional representation of the ME filename convention.

    Filename structure (dot-separated):
        location_id.dataset_name[-qualifier][-temporal].data_level.date.time[.v{version}][.created=timestamp].ext

    Components at fixed positions from the start:
        0: location_id      (e.g., "AK_cook_inlet")
        1: dataset_name     (e.g., "wpto_high_res_tidal-year_average")
        2: data_level       (e.g., "b3")
        3: date             (YYYYMMDD, e.g., "20100603") — optional
        4: time             (HHMMSS, e.g., "000000") — optional
        tail: version, creation timestamp, extension — all optional
    """

    location_id: str
    dataset_name: str  # includes qualifier and temporal (hyphen-joined)
    data_level: str
    date: str = None  # YYYYMMDD
    time: str = None  # HHMMSS
    version: str = None  # e.g., "1.0.0" (without "v" prefix)
    creation_timestamp: str = None  # e.g., "20260205_1430Z"
    ext: str = None

    @classmethod
    def from_filename(cls, filename):
        """Parse a filename (with or without path) into its components.

        Examples:
            >>> DataFileName.from_filename("AK_cook_inlet.wpto_high_res_tidal-year_average.b3.20100603.000000.v1.0.0.nc")
            DataFileName(location_id='AK_cook_inlet', dataset_name='wpto_high_res_tidal-year_average',
                        data_level='b3', date='20100603', time='000000', version='1.0.0', ext='nc')
        """
        name = Path(str(filename)).name
        parts = name.split(".")

        location_id = parts[0]
        dataset_name = parts[1]
        data_level = parts[2]

        # Detect whether date/time are present (8-digit date at position 3)
        date = None
        time = None
        tail_start = 3

        if len(parts) > 4 and len(parts[3]) == 8 and parts[3].isdigit():
            date = parts[3]
            time = parts[4]
            tail_start = 5

        # Parse remaining parts: version, creation timestamp, extension
        remaining = parts[tail_start:]
        version = None
        creation_timestamp = None
        ext = None

        i = 0
        while i < len(remaining):
            part = remaining[i]
            if part.startswith("v") and len(part) > 1 and part[1:].isdigit():
                # Version start — collect "v1", "0", "0" style dot-split parts
                version_parts = [part[1:]]
                i += 1
                while i < len(remaining) and remaining[i].isdigit():
                    version_parts.append(remaining[i])
                    i += 1
                version = ".".join(version_parts)
            elif part.startswith("created="):
                creation_timestamp = part[len("created="):]
                i += 1
            else:
                # File extension (last non-version, non-timestamp part)
                ext = part
                i += 1

        return cls(
            location_id=location_id,
            dataset_name=dataset_name,
            data_level=data_level,
            date=date,
            time=time,
            version=version,
            creation_timestamp=creation_timestamp,
            ext=ext,
        )

    def to_filename(self):
        """Reconstruct the filename string from components."""
        components = [self.location_id, self.dataset_name, self.data_level]

        if self.date is not None:
            components.extend([self.date, self.time])

        if self.version is not None:
            components.append(f"v{self.version}")

        if self.creation_timestamp is not None:
            components.append(f"created={self.creation_timestamp}")

        if self.ext is not None:
            components.append(self.ext)

        return ".".join(components)


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
    version=None,
    include_creation_timestamp=None,
    include_dataset_timestamp=True,
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
        version: Optional version string (e.g., '1.0', '2.1')
        include_timestamp: Boolean flag to include creation timestamp
    Returns:
        Formatted filename string: location_id.dataset_name[-qualifier][-temporal].data_level.date.time[.v{version}][.timestamp].ext
    """

    components = [location_id, dataset_name]

    if qualifier:
        components[-1] += f"-{qualifier}"
    if temporal:
        components[-1] += f"-{temporal}"

    components.append(data_level)

    if include_dataset_timestamp is True:
        if static_time is None:
            date_str, time_str = _format_datetime(ds)
        else:
            date_str, time_str = static_time[0], static_time[1]

        components.extend([date_str, time_str])

    # Add version if provided
    if version:
        components.append(f"v{version}")

    # Generate and add timestamp if requested
    if include_creation_timestamp:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%MZ")
        components.append(f"created={timestamp}")

    if ext is not None:
        return ".".join(components + [ext])
    else:
        return ".".join(components)


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
