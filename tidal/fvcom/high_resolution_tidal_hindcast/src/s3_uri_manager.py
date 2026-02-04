"""
S3 URI Manager for Marine Energy Data

This module provides utilities for generating S3 URIs and HTTPS URLs
for accessing marine energy data files stored in the AWS Open Data registry.

The primary use case is generating `full_year_data_s3_uri` and `full_year_data_https_url`
columns for b4/b5 summary parquet files that link to the corresponding b1 full-year
time series parquet files.

S3 Access Methods:
- S3 URI: s3://marine-energy-data/us-tidal/... (for AWS CLI, boto3, pandas)
- HTTPS URL: https://marine-energy-data.s3.us-west-2.amazonaws.com/... (for browsers, curl)

Example S3 URI:
    s3://marine-energy-data/us-tidal/AK_cook_inlet/v1.0.0/b1_vap_by_point_partition/
    lat_deg=59/lon_deg=-152/lat_dec=12/lon_dec=78/
    AK_cook_inlet.wpto_high_res_tidal.face=00012345.lat=59.1234567.lon=-152.7890123-1h.b1.20050101.000000.v1.0.0.parquet
"""

from datetime import datetime

import numpy as np
import pandas as pd


# AWS region for the marine-energy-data bucket (used in HTTPS URLs)
AWS_REGION = "us-west-2"


def get_s3_base_uri(config):
    """
    Get the S3 base URI from configuration.

    Returns:
        str: S3 URI like 's3://marine-energy-data/us-tidal'
    """
    bucket = config["storage"]["s3_bucket"]
    prefix = config["storage"]["s3_prefix"]
    return f"s3://{bucket}/{prefix}"


def get_https_base_url(config):
    """
    Get the HTTPS base URL for direct browser/download access.

    The format is: https://{bucket}.s3.{region}.amazonaws.com/{prefix}

    Returns:
        str: HTTPS URL like 'https://marine-energy-data.s3.us-west-2.amazonaws.com/us-tidal'
    """
    bucket = config["storage"]["s3_bucket"]
    prefix = config["storage"]["s3_prefix"]
    return f"https://{bucket}.s3.{AWS_REGION}.amazonaws.com/{prefix}"


def get_b1_partition_path(lat, lon, config):
    """
    Generate the partition directory path based on lat/lon coordinates.

    The partition structure is: lat_deg={DD}/lon_deg={DD}/lat_dec={DD}/lon_dec={DD}/

    Args:
        lat: Latitude value (float)
        lon: Longitude value (float)
        config: Configuration dictionary

    Returns:
        str: Partition path like 'lat_deg=59/lon_deg=-152/lat_dec=12/lon_dec=78'
    """
    decimal_places = config["partition"]["decimal_places"]

    lat_deg = int(lat)
    lon_deg = int(lon)

    # Extract decimal portion based on decimal_places config
    multiplier = 10 ** decimal_places
    lat_dec = int(abs(lat * multiplier) % multiplier)
    lon_dec = int(abs(lon * multiplier) % multiplier)

    # Format width matches decimal_places
    format_spec = f"0{decimal_places}d"

    return f"lat_deg={lat_deg}/lon_deg={lon_deg}/lat_dec={lat_dec:{format_spec}}/lon_dec={lon_dec:{format_spec}}"


def get_b1_filename(face_id, lat, lon, location, config):
    """
    Generate the b1 parquet filename for a given face.

    Filename format:
    {location}.{dataset_name}.face={face_id}.lat={lat}.lon={lon}-{temporal}.b1.{date}.{time}.v{version}.parquet

    Args:
        face_id: Face ID (int or str)
        lat: Latitude value (float)
        lon: Longitude value (float)
        location: Location configuration dictionary
        config: Configuration dictionary

    Returns:
        str: Filename like 'AK_cook_inlet.wpto_high_res_tidal.face=00012345.lat=59.1234567.lon=-152.7890123-1h.b1.20050101.000000.v1.0.0.parquet'
    """
    coord_digits_max = config["partition"]["coord_digits_max"]
    index_max_digits = config["partition"]["index_max_digits"]
    version = config["dataset"]["version"]
    dataset_name = config["dataset"]["name"]
    output_name = location["output_name"]

    # Determine temporal string from delta_t
    expected_delta_t_seconds = location["expected_delta_t_seconds"]
    temporal_mapping = {3600: "1h", 1800: "30m"}
    if expected_delta_t_seconds not in temporal_mapping:
        raise ValueError(
            f"Unexpected expected_delta_t_seconds: {expected_delta_t_seconds}"
        )
    temporal_string = temporal_mapping[expected_delta_t_seconds]

    # Get start date from location config for the filename timestamp
    start_date_str = location["start_date_utc"]
    start_dt = pd.Timestamp(start_date_str)
    date_str = start_dt.strftime("%Y%m%d")
    time_str = start_dt.strftime("%H%M%S")

    # Format face_id with zero-padding
    if isinstance(face_id, str):
        # If already a string (e.g., "00012345"), extract the numeric part
        face_id_int = int(face_id)
    else:
        face_id_int = int(face_id)

    # Format coordinates with specified precision
    lat_rounded = round(float(lat), coord_digits_max)
    lon_rounded = round(float(lon), coord_digits_max)

    index_format = f"0{index_max_digits}d"
    coord_format = f".{coord_digits_max}f"

    return (
        f"{output_name}.{dataset_name}."
        f"face={face_id_int:{index_format}}."
        f"lat={lat_rounded:{coord_format}}."
        f"lon={lon_rounded:{coord_format}}-{temporal_string}."
        f"b1.{date_str}.{time_str}.v{version}.parquet"
    )


def get_b1_relative_path(face_id, lat, lon, location, config):
    """
    Generate the full relative path (from S3 prefix) to a b1 parquet file.

    Path structure:
    {location}/{version}/b1_vap_by_point_partition/{partition_path}/{filename}

    Args:
        face_id: Face ID (int or str)
        lat: Latitude value (float)
        lon: Longitude value (float)
        location: Location configuration dictionary
        config: Configuration dictionary

    Returns:
        str: Relative path from S3 prefix
    """
    output_name = location["output_name"]
    version = f"v{config['dataset']['version']}"

    partition_path = get_b1_partition_path(lat, lon, config)
    filename = get_b1_filename(face_id, lat, lon, location, config)

    return f"{output_name}/{version}/b1_vap_by_point_partition/{partition_path}/{filename}"


def get_full_year_s3_uri(face_id, lat, lon, location, config):
    """
    Generate the complete S3 URI for a face's full-year b1 parquet file.

    Args:
        face_id: Face ID (int or str)
        lat: Latitude value (float)
        lon: Longitude value (float)
        location: Location configuration dictionary
        config: Configuration dictionary

    Returns:
        str: Complete S3 URI

    Example:
        's3://marine-energy-data/us-tidal/AK_cook_inlet/v1.0.0/b1_vap_by_point_partition/...'
    """
    base_uri = get_s3_base_uri(config)
    relative_path = get_b1_relative_path(face_id, lat, lon, location, config)
    return f"{base_uri}/{relative_path}"


def get_full_year_https_url(face_id, lat, lon, location, config):
    """
    Generate the complete HTTPS URL for a face's full-year b1 parquet file.

    This URL can be used directly in browsers or with curl/wget for downloading.

    Args:
        face_id: Face ID (int or str)
        lat: Latitude value (float)
        lon: Longitude value (float)
        location: Location configuration dictionary
        config: Configuration dictionary

    Returns:
        str: Complete HTTPS URL

    Example:
        'https://marine-energy-data.s3.us-west-2.amazonaws.com/us-tidal/AK_cook_inlet/v1.0.0/...'
    """
    base_url = get_https_base_url(config)
    relative_path = get_b1_relative_path(face_id, lat, lon, location, config)
    return f"{base_url}/{relative_path}"


def add_full_year_uri_columns(df, location, config):
    """
    Add full_year_data_s3_uri and full_year_data_https_url columns to a DataFrame.

    This function is designed for use with b4/b5 summary parquet files that contain
    face_id, lat_center, and lon_center columns.

    Args:
        df: DataFrame with face_id, lat_center, lon_center columns
        location: Location configuration dictionary
        config: Configuration dictionary

    Returns:
        DataFrame with added URI columns

    Note:
        The DataFrame must have either:
        - 'lat_center' and 'lon_center' columns, OR
        - 'lat' and 'lon' columns (for atlas subset format)
    """
    # Determine column names for lat/lon
    if "lat_center" in df.columns:
        lat_col = "lat_center"
        lon_col = "lon_center"
    elif "lat" in df.columns:
        lat_col = "lat"
        lon_col = "lon"
    else:
        raise ValueError("DataFrame must have lat_center/lon_center or lat/lon columns")

    # Vectorized generation of URIs
    # For performance with large DataFrames, we build the URIs using vectorized string operations

    s3_uris = []
    https_urls = []

    for idx, row in df.iterrows():
        face_id = row["face_id"]
        lat = row[lat_col]
        lon = row[lon_col]

        s3_uri = get_full_year_s3_uri(face_id, lat, lon, location, config)
        https_url = get_full_year_https_url(face_id, lat, lon, location, config)

        s3_uris.append(s3_uri)
        https_urls.append(https_url)

    df = df.copy()
    df["full_year_data_s3_uri"] = s3_uris
    df["full_year_data_https_url"] = https_urls

    return df


def add_full_year_uri_columns_vectorized(df, location, config):
    """
    Vectorized version of add_full_year_uri_columns for better performance.

    Uses NumPy/Pandas vectorized operations instead of row-by-row iteration.
    Recommended for DataFrames with >10,000 rows.

    Args:
        df: DataFrame with face_id, lat_center, lon_center columns
        location: Location configuration dictionary
        config: Configuration dictionary

    Returns:
        DataFrame with added URI columns
    """
    # Determine column names for lat/lon
    if "lat_center" in df.columns:
        lat_col = "lat_center"
        lon_col = "lon_center"
    elif "lat" in df.columns:
        lat_col = "lat"
        lon_col = "lon"
    else:
        raise ValueError("DataFrame must have lat_center/lon_center or lat/lon columns")

    # Extract config values once
    decimal_places = config["partition"]["decimal_places"]
    coord_digits_max = config["partition"]["coord_digits_max"]
    index_max_digits = config["partition"]["index_max_digits"]
    version = config["dataset"]["version"]
    dataset_name = config["dataset"]["name"]
    output_name = location["output_name"]

    # Temporal string
    expected_delta_t_seconds = location["expected_delta_t_seconds"]
    temporal_mapping = {3600: "1h", 1800: "30m"}
    temporal_string = temporal_mapping[expected_delta_t_seconds]

    # Date/time from location start
    start_dt = pd.Timestamp(location["start_date_utc"])
    date_str = start_dt.strftime("%Y%m%d")
    time_str = start_dt.strftime("%H%M%S")

    # Base URIs
    s3_base = get_s3_base_uri(config)
    https_base = get_https_base_url(config)

    # Partition path components
    lats = df[lat_col].values
    lons = df[lon_col].values

    lat_deg = lats.astype(int)
    lon_deg = lons.astype(int)

    multiplier = 10 ** decimal_places
    lat_dec = (np.abs(lats * multiplier) % multiplier).astype(int)
    lon_dec = (np.abs(lons * multiplier) % multiplier).astype(int)

    # Format coordinates
    lat_rounded = np.round(lats, coord_digits_max)
    lon_rounded = np.round(lons, coord_digits_max)

    # Build path components as Series for vectorized string concatenation
    dec_format = f"0{decimal_places}d"
    coord_format = f".{coord_digits_max}f"
    index_format = f"0{index_max_digits}d"

    # For face_id - handle both string and int types
    if df["face_id"].dtype == object:  # string type
        face_ids = df["face_id"].astype(int)
    else:
        face_ids = df["face_id"]

    # Build partition paths
    partition_paths = (
        "lat_deg=" + pd.Series(lat_deg).apply(lambda x: str(x)) +
        "/lon_deg=" + pd.Series(lon_deg).apply(lambda x: str(x)) +
        "/lat_dec=" + pd.Series(lat_dec).apply(lambda x: f"{x:{dec_format}}") +
        "/lon_dec=" + pd.Series(lon_dec).apply(lambda x: f"{x:{dec_format}}")
    )

    # Build filenames
    filenames = (
        f"{output_name}.{dataset_name}." +
        "face=" + pd.Series(face_ids).apply(lambda x: f"{x:{index_format}}") +
        ".lat=" + pd.Series(lat_rounded).apply(lambda x: f"{x:{coord_format}}") +
        ".lon=" + pd.Series(lon_rounded).apply(lambda x: f"{x:{coord_format}}") +
        f"-{temporal_string}.b1.{date_str}.{time_str}.v{version}.parquet"
    )

    # Build full paths
    relative_paths = (
        f"{output_name}/v{version}/b1_vap_by_point_partition/" +
        partition_paths + "/" + filenames
    )

    df = df.copy()
    df["full_year_data_s3_uri"] = s3_base + "/" + relative_paths
    df["full_year_data_https_url"] = https_base + "/" + relative_paths

    return df