from pathlib import Path
import os
import time
import random
import gc
from datetime import datetime
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import xarray as xr

from . import file_manager, file_name_convention_manager

# Number of threads for parallel parquet file writes
# Adjust based on HPC filesystem capabilities (Lustre can handle higher values)
PARQUET_WRITE_THREADS = 16

# Number of threads for parallel HDF5 variable reads within a single file
# HDF5 supports concurrent reads from different datasets
HDF5_READ_THREADS = 8

# Stagger job start times to reduce Lustre I/O contention (seconds)
# When many jobs start simultaneously, they compete for the same files
JOB_STAGGER_MAX_SECONDS = 120


def prepare_nc_metadata_for_parquet(attributes):
    """
    This outputs two dictionaries of atttributes that are compatiable with parquet metadata
    The `global` dict is equivalent to the global attributes in a netCDF file,
    The `vars` dict is a map of variable names and is equivalent to the variable attributes in a netCDF file.
    """

    # Custom JSON encoder for NumPy types
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
                return float(obj)
            elif isinstance(obj, (np.ndarray,)):
                return obj.tolist()
            elif isinstance(obj, (np.bool_)):
                return bool(obj)
            elif np.isnan(obj):
                return None
            elif hasattr(obj, "isoformat"):  # datetime objects
                return obj.isoformat()
            return super(NumpyEncoder, self).default(obj)

    global_attrs = {}
    variable_attrs = {}

    # Set identifiers in metadata
    global_attrs["WPTO_HINDCAST_FORMAT_VERSION"] = "1.0"
    global_attrs["WPTO_HINDCAST_METADATA_TYPE"] = "netcdf_compatible"

    global_attrs_to_skip = [
        # This does not provide relevant information here
        "input_history_json",
        "geospatial_bounds",
        "geospatial_bounds_crs",
        # global:geospatial_bounds_crs: +proj=latlon +datum=WGS84 +ellps=WGS84 +type=crs
        # global:geospatial_lat_max: 61.499656677246094
        "geospatial_lat_max",
        # global:geospatial_lat_min: 58.90425491333008
        "geospatial_lat_min",
        # global:geospatial_lat_units: degrees_north
        # global:geospatial_lon_max: -148.990478515625
        "geospatial_lon_max",
        # global:geospatial_lon_min: -154.2655029296875
        "geospatial_lon_min",
        # global:geospatial_lon_units: degrees_east
        # global:geospatial_vertical_max: 171.5044708251953
        "geospatial_vertical_max",
        # global:geospatial_vertical_min: -6.143433570861816
        "geospatial_vertical_min",
        # global:geospatial_vertical_origin: geoid
        # global:geospatial_vertical_positive: down
        # global:geospatial_vertical_units: m
    ]
    for attr_name, attr_value in attributes["global"].items():
        if attr_name not in global_attrs_to_skip:
            global_attrs[attr_name] = attr_value

    for var_name, var_attrs in attributes["vars"].items():
        if var_name not in variable_attrs:
            variable_attrs[var_name] = {}
        for attr_name, attr_value in var_attrs.items():
            variable_attrs[var_name][attr_name] = attr_value

    # Convert all metadata values to bytes
    global_attr_bytes = {}
    for k, v in global_attrs.items():
        global_attr_bytes[k] = str(v).encode("utf-8")

    variable_attr_bytes = {}
    for key in variable_attrs.keys():
        variable_attr_bytes[key] = {}
        for k, v in variable_attrs[key].items():
            variable_attr_bytes[key][k] = str(v).encode("utf-8")

    return {
        "global": global_attr_bytes,
        "var": variable_attr_bytes,
    }


def get_partition_path(df, config) -> str:
    """
    Generate the partition path based on lat/lon coordinates.
    """
    coordinate_decimal_places = config["partition"]["decimal_places"]
    lat = df["lat"].iloc[0]
    lon = df["lon"].iloc[0]

    lat_deg = int(lat)
    lon_deg = int(lon)

    # Extract decimal decimal_places based on coordinate_decimal_places
    multiplier = 10**coordinate_decimal_places
    lat_dec = int(abs(lat * multiplier) % multiplier)
    lon_dec = int(abs(lon * multiplier) % multiplier)

    # Use coordinate_decimal_places for formatting width
    format_spec = f"0{coordinate_decimal_places}d"

    return f"lat_deg={lat_deg}/lon_deg={lon_deg}/lat_dec={lat_dec:{format_spec}}/lon_dec={lon_dec:{format_spec}}"


def get_partition_file_name(
    index: int,
    df,
    config,
    location,
) -> str:
    """
    Generate standardized filename for partition files.
    """

    coord_digits_max = config["partition"]["coord_digits_max"]
    index_max_digits = config["partition"]["index_max_digits"]
    version = config["dataset"]["version"]

    # Round latitude and longitude to specified decimal places
    lat_rounded = round(df["lat"].iloc[0], coord_digits_max)
    lon_rounded = round(df["lon"].iloc[0], coord_digits_max)

    # Determine temporal string based on expected_delta_t_seconds
    expected_delta_t_seconds = location["expected_delta_t_seconds"]
    temporal_mapping = {3600: "1h", 1800: "30m"}
    if expected_delta_t_seconds not in temporal_mapping:
        raise ValueError(
            f"Unexpected expected_delta_t_seconds configuration {expected_delta_t_seconds}"
        )
    temporal_string = temporal_mapping[expected_delta_t_seconds]

    # Format strings for padding and precision
    index_format = f"0{index_max_digits}d"
    coord_format = f".{coord_digits_max}f"

    # Use file name convention manager to generate standard filename
    return file_name_convention_manager.generate_filename_for_data_level(
        df,
        location["output_name"],
        f"{config['dataset']['name']}.face={index:{index_format}}.lat={lat_rounded:{coord_format}}.lon={lon_rounded:{coord_format}}",
        "b4",
        temporal=temporal_string,
        version=version,
        ext="parquet",
    )


def get_dataset_info(h5_file_path):
    """
    Get information about datasets in the h5 file

    Parameters:
    -----------
    h5_file_path : str
        Path to the h5 file

    Returns:
    --------
    dict
        Dictionary containing dataset information
    """
    dataset_info = {
        "total_faces": 0,
        "2d_datasets": [],  # (time, face)
        "3d_datasets": [],  # (time, layer, face)
        "other_datasets": [],
        "time_length": 0,
    }

    try:
        with h5py.File(h5_file_path, "r") as f:
            # Get the number of faces
            if "lat_center" in f:
                dataset_info["total_faces"] = f["lat_center"].shape[0]
            elif "lon_center" in f:
                dataset_info["total_faces"] = f["lon_center"].shape[0]
            else:
                raise ValueError(
                    "Could not determine number of faces from lat_center or lon_center"
                )

            # Get time length
            if "time" in f:
                dataset_info["time_length"] = f["time"].shape[0]

            # Categorize datasets
            for dataset_name, dataset in f.items():
                if isinstance(dataset, h5py.Dataset):
                    shape = dataset.shape

                    # Skip lat/lon nodes as they're not per-face
                    if dataset_name in [
                        "lat_node",
                        "lon_node",
                        "node",
                        "face_node",
                        "face_node_index",
                    ]:
                        continue

                    # Skip single-dimension datasets that aren't time
                    if len(shape) == 1 and dataset_name != "time":
                        continue

                    # 2D datasets with (time, face)
                    if len(shape) == 2 and shape[1] == dataset_info["total_faces"]:
                        dataset_info["2d_datasets"].append(dataset_name)

                    # 3D datasets with (time, layer, face)
                    elif len(shape) == 3 and shape[2] == dataset_info["total_faces"]:
                        dataset_info["3d_datasets"].append(
                            (dataset_name, shape[1])
                        )  # Include number of layers

                    # Other datasets
                    else:
                        dataset_info["other_datasets"].append(dataset_name)

    except Exception as e:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(
            f"{timestamp} - ERROR - Error getting dataset info from {h5_file_path}: {e}"
        )
        raise

    return dataset_info


def extract_metadata_from_nc(nc_file_path):
    """
    Extract metadata from the first NC file.

    Parameters:
    -----------
    nc_file_path : str
        Path to the netCDF file

    Returns:
    --------
    dict
        Dictionary containing prepared metadata
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{timestamp} - INFO - Extracting metadata from {nc_file_path}")

    attrs = {}
    with xr.open_dataset(nc_file_path) as ds:
        global_attrs = dict(ds.attrs)

        # Get variable attributes
        var_attrs = {}
        for var_name, var in ds.variables.items():
            var_attrs[var_name] = dict(var.attrs)

        attrs = {"global": global_attrs, "vars": var_attrs}

    #     print("Passing metadata to prepare_nc_metadata_for_parquet")
    #
    # for key, value in attrs["vars"].items():
    #     print(f"Variable: {key}")
    #     for attr_key, attr_value in value.items():
    #         print(f"  {attr_key}: {attr_value}")

    return prepare_nc_metadata_for_parquet(attrs)


def build_parquet_schema_with_metadata(
    column_names,
    column_dtypes,
    nc_metadata_for_parquet,
    parquet_col_to_nc_var_map,
):
    """
    Build a PyArrow schema with metadata pre-computed.

    This function creates the schema ONCE before the write loop, avoiding
    redundant schema reconstruction for every file.

    Parameters:
    -----------
    column_names : list
        List of column names in order
    column_dtypes : dict
        Mapping of column name to PyArrow data type
    nc_metadata_for_parquet : dict
        Metadata dict with 'global' and 'var' keys
    parquet_col_to_nc_var_map : dict
        Mapping from parquet column names to NC variable names

    Returns:
    --------
    pa.Schema
        PyArrow schema with all field and file metadata attached
    """
    fields = []

    for col_name in column_names:
        # Get the PyArrow type for this column
        pa_type = column_dtypes.get(col_name, pa.float32())

        # Build field metadata from NC variable attributes
        field_metadata = {}
        nc_var_name = parquet_col_to_nc_var_map.get(col_name, col_name)
        if nc_var_name in nc_metadata_for_parquet["var"]:
            field_metadata = nc_metadata_for_parquet["var"][nc_var_name].copy()

        # Create the field with metadata
        field = pa.field(
            col_name, pa_type, metadata=field_metadata if field_metadata else None
        )
        fields.append(field)

    # Create schema and attach global metadata
    schema = pa.schema(fields)
    schema = schema.with_metadata(nc_metadata_for_parquet.get("global", {}))

    return schema


def write_parquet_file_safe(args):
    """
    Write a single parquet file with error handling.

    Parameters:
    -----------
    args : tuple
        (face_id, table, output_path)

    Returns:
    --------
    tuple
        (face_id, output_path, error_or_None)
    """
    face_id, table, output_path = args
    try:
        # Ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, output_path)
        return (face_id, str(output_path), None)
    except Exception as e:
        return (face_id, str(output_path), str(e))


def get_partition_path_from_coords(lat, lon, config):
    """
    Generate the partition path based on lat/lon coordinates.

    This version takes scalar values directly instead of a DataFrame,
    avoiding pandas overhead.
    """
    coordinate_decimal_places = config["partition"]["decimal_places"]

    lat_deg = int(lat)
    lon_deg = int(lon)

    # Extract decimal places based on coordinate_decimal_places
    multiplier = 10**coordinate_decimal_places
    lat_dec = int(abs(lat * multiplier) % multiplier)
    lon_dec = int(abs(lon * multiplier) % multiplier)

    # Use coordinate_decimal_places for formatting width
    format_spec = f"0{coordinate_decimal_places}d"

    return f"lat_deg={lat_deg}/lon_deg={lon_deg}/lat_dec={lat_dec:{format_spec}}/lon_dec={lon_dec:{format_spec}}"


def get_partition_file_name_from_coords(
    index: int,
    lat: float,
    lon: float,
    time_values,
    config,
    location,
) -> str:
    """
    Generate standardized filename for partition files.

    This version takes scalar values directly instead of a DataFrame,
    using file_name_convention_manager with static_time parameter.
    """
    coord_digits_max = config["partition"]["coord_digits_max"]
    index_max_digits = config["partition"]["index_max_digits"]
    version = config["dataset"]["version"]

    # Round latitude and longitude to specified decimal places
    lat_rounded = round(lat, coord_digits_max)
    lon_rounded = round(lon, coord_digits_max)

    # Determine temporal string based on expected_delta_t_seconds
    expected_delta_t_seconds = location["expected_delta_t_seconds"]
    temporal_mapping = {3600: "1h", 1800: "30m"}
    if expected_delta_t_seconds not in temporal_mapping:
        raise ValueError(
            f"Unexpected expected_delta_t_seconds configuration {expected_delta_t_seconds}"
        )
    temporal_string = temporal_mapping[expected_delta_t_seconds]

    # Format strings for padding and precision
    index_format = f"0{index_max_digits}d"
    coord_format = f".{coord_digits_max}f"

    # Get time range for filename (first timestamp)
    first_time = pd.Timestamp(time_values[0])
    date_str = first_time.strftime("%Y%m%d")
    time_str = first_time.strftime("%H%M%S")

    # Use file_name_convention_manager as single source of truth
    # Pass static_time to avoid needing a DataFrame
    return file_name_convention_manager.generate_filename_for_data_level(
        ds=None,  # Not needed when static_time is provided
        location_id=location["output_name"],
        dataset_name=f"{config['dataset']['name']}.face={index:{index_format}}.lat={lat_rounded:{coord_format}}.lon={lon_rounded:{coord_format}}",
        data_level="b4",
        temporal=temporal_string,
        version=version,
        ext="parquet",
        static_time=(date_str, time_str),
    )


def convert_h5_to_parquet_batched(
    input_dir,
    output_dir,
    config,
    location,
    batch_size=20000,
    batch_number=0,
    skip_existing=False,
):
    """
    Convert h5 files to individual parquet files for each face using a sequential approach.

    This approach:
    1. Sorts input files by name (assumes time-ordered, e.g., monthly files)
    2. Uses batch reading for each variable (reading all faces in batch at once)
    3. Combines data into a single DataFrame with all time points (e.g., a full year)
    4. Writes one parquet file per face with the complete time series
    5. Includes element corner coordinates from the nv, lat_node, and lon_node datasets
    6. Organizes files in partitioned directory structure based on lat/lon
    7. Extracts and includes metadata from the first NC file

    Parameters:
    -----------
    input_dir : str
        Directory containing h5 files
    output_dir : str
        Directory to save parquet files
    config : dict
        Configuration dictionary
    location : dict
        Location configuration
    batch_size : int
        Number of faces to process in each batch
    batch_number : int
        Batch number to process
    skip_existing : bool
        Skip processing faces whose output files already exist
    """
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(
        f"{timestamp} - INFO - Starting h5 to parquet conversion with optimized sequential approach"
    )

    # Stagger job start to reduce Lustre I/O contention
    # When many jobs start simultaneously, they all compete for the same files
    stagger_delay = random.uniform(0, JOB_STAGGER_MAX_SECONDS)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(
        f"{timestamp} - INFO - Staggering job start by {stagger_delay:.1f} seconds to reduce I/O contention"
    )
    time.sleep(stagger_delay)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get all h5 files and sort them by name (assumes chronological ordering)
    h5_files = sorted(list(Path(input_dir).glob("*.nc")))
    if not h5_files:
        h5_files = sorted(list(Path(input_dir).glob("*.h5")))
        if not h5_files:
            print(f"{timestamp} - ERROR - No h5 or nc files found in {input_dir}")
            return

    print(f"{timestamp} - INFO - Found {len(h5_files)} files (sorted by name)")
    for i, file in enumerate(h5_files):
        print(f"  File {i + 1}: {file.name}")

    # Get dataset information from the first file
    print(f"{timestamp} - INFO - Reading dataset information from first file")
    dataset_info = get_dataset_info(h5_files[0])
    total_faces = dataset_info["total_faces"]

    # Extract metadata from the first NC file
    nc_metadata_for_parquet = extract_metadata_from_nc(h5_files[0])

    print(f"{timestamp} - INFO - Extracted metadata from {h5_files[0]}")

    print(f"{timestamp} - INFO - Total faces to process: {total_faces}")
    print(f"{timestamp} - INFO - 2D datasets: {dataset_info['2d_datasets']}")
    print(
        f"{timestamp} - INFO - 3D datasets: {[(name, layers) for name, layers in dataset_info['3d_datasets']]}"
    )

    # First, read the node indices and coordinates from the first file
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{timestamp} - INFO - Reading element corner coordinates")

    # Dictionary to store node indices and coordinates for all faces
    element_corners = {}

    with h5py.File(h5_files[0], "r") as f:
        print(f"{timestamp} - INFO - Reading lat_node and lon_node arrays")
        # Read node coordinates
        lat_node = f["lat_node"][:]
        lon_node = f["lon_node"][:]
        print(f"{timestamp} - INFO - Read {len(lat_node)} node coordinates")

        print(f"{timestamp} - INFO - Reading node indices (nv array)")
        # Read node indices for each face (first time step is sufficient)
        # Note: nv is (3, face) where 3 is the number of corners in each triangular face
        nv = f["nv"][:, :]
        print(f"{timestamp} - INFO - nv shape: {nv.shape}")

        # Convert from Fortran 1-based indexing to Python 0-based indexing
        print(f"{timestamp} - INFO - Converting from 1-based to 0-based indexing")
        # Ensure we have a numpy array and convert to 0-based indexing
        nv = np.array(nv) - 1
        print(
            f"{timestamp} - INFO - nv min/max after conversion: {nv.min()}/{nv.max()}"
        )

        # For each face, get the corner coordinates
        print(f"{timestamp} - INFO - Calculating corner coordinates for all faces")
        for face_id in range(total_faces):
            if face_id % 100000 == 0 and face_id > 0:
                print(
                    f"{timestamp} - INFO - Processed corner coordinates for {face_id} faces"
                )

            # Get the 3 node indices for this face
            node_indices = nv[:, face_id]

            # Get the coordinates for each corner
            element_corners[face_id] = {
                "element_corner_1_lat": lat_node[node_indices[0]],
                "element_corner_1_lon": lon_node[node_indices[0]],
                "element_corner_2_lat": lat_node[node_indices[1]],
                "element_corner_2_lon": lon_node[node_indices[1]],
                "element_corner_3_lat": lat_node[node_indices[2]],
                "element_corner_3_lon": lon_node[node_indices[2]],
            }
        print(
            f"{timestamp} - INFO - Completed corner coordinates for all {total_faces} faces"
        )

    # Process faces in batches
    start_index = batch_number * batch_size
    start_face = start_index
    print(f"{timestamp} - INFO - Processing {batch_size} faces in batch {batch_number}")

    batch_start_time = time.time()
    end_face = min(start_face + batch_size, total_faces)
    faces_to_process = end_face - start_face

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Data structure to store all time series for each face
    all_face_data = {}

    parquet_col_to_nc_var_map = {}

    # Direct mappings for coordinate transformations
    parquet_col_to_nc_var_map["lat"] = "lat_center"  # lat_center -> lat
    parquet_col_to_nc_var_map["lon"] = "lon_center"  # lon_center -> lon

    # Element corner coordinates (these are computed, no direct NC equivalent)
    # We could map these to lat_node/lon_node if that makes sense
    parquet_col_to_nc_var_map["element_corner_1_lat"] = "lat_node"
    parquet_col_to_nc_var_map["element_corner_1_lon"] = "lon_node"
    parquet_col_to_nc_var_map["element_corner_2_lat"] = "lat_node"
    parquet_col_to_nc_var_map["element_corner_2_lon"] = "lon_node"
    parquet_col_to_nc_var_map["element_corner_3_lat"] = "lat_node"
    parquet_col_to_nc_var_map["element_corner_3_lon"] = "lon_node"

    # 2D datasets - direct mapping
    for dataset_name in dataset_info["2d_datasets"]:
        if dataset_name not in ["lat_center", "lon_center"]:
            parquet_col_to_nc_var_map[dataset_name] = dataset_name

    # 3D datasets - layer expansion mapping
    for dataset_name, num_layers in dataset_info["3d_datasets"]:
        for layer_idx in range(num_layers):
            parquet_col = f"{dataset_name}_layer_{layer_idx}"
            parquet_col_to_nc_var_map[parquet_col] = (
                dataset_name  # All layers map back to original variable
            )

    # Time mapping
    parquet_col_to_nc_var_map["time"] = "time"

    # Initialize the data structure for each face
    print(
        f"{timestamp} - INFO - Initializing data structure for {faces_to_process} faces"
    )
    for face_id in range(start_face, end_face):
        all_face_data[face_id] = {
            "time": [],
            "lat": None,
            "lon": None,
            # Element corner coordinates (constant)
            "element_corner_1_lat": element_corners[face_id]["element_corner_1_lat"],
            "element_corner_1_lon": element_corners[face_id]["element_corner_1_lon"],
            "element_corner_2_lat": element_corners[face_id]["element_corner_2_lat"],
            "element_corner_2_lon": element_corners[face_id]["element_corner_2_lon"],
            "element_corner_3_lat": element_corners[face_id]["element_corner_3_lat"],
            "element_corner_3_lon": element_corners[face_id]["element_corner_3_lon"],
        }

        # Initialize dataset arrays
        for dataset_name in dataset_info["2d_datasets"]:
            if dataset_name not in ["lat_center", "lon_center"]:
                all_face_data[face_id][dataset_name] = []

        for dataset_name, num_layers in dataset_info["3d_datasets"]:
            for layer_idx in range(num_layers):
                col_name = f"{dataset_name}_layer_{layer_idx}"
                all_face_data[face_id][col_name] = []

    # Get HDF5 read cache size from config (default to 2GB if not configured)
    hdf5_read_cache_bytes = config.get("hdf5_cache", {}).get("read_cache_bytes")
    if hdf5_read_cache_bytes is None:
        hdf5_read_cache_bytes = 2 * 1024**3  # 2GB default
    # Use prime number for cache slots to reduce hash collisions
    # From the https://docs.h5py.org/en/stable/high/file.html#chunk-cache:
    #
    # rdcc_nslots: Number of chunk slots in the raw data chunk cache for files opened with this property list.
    # Default is 521. Increasing this value reduces the number of cache collisions, but slightly increases the
    # memory used. A prime number is recommended.

    hdf5_cache_slots = 10007

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(
        f"{timestamp} - INFO - Using HDF5 read cache: {hdf5_read_cache_bytes / 1024**3:.1f} GB"
    )

    # Process each file sequentially
    for file_idx, h5_file in enumerate(h5_files):
        file_start_time = time.time()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(
            f"{timestamp} - INFO - File {file_idx + 1}/{len(h5_files)}: Reading {h5_file}"
        )

        # Open with configured HDF5 chunk cache for better read performance
        with h5py.File(
            h5_file,
            "r",
            rdcc_nbytes=hdf5_read_cache_bytes,
            rdcc_nslots=hdf5_cache_slots,
        ) as f:
            # Get time values once
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"{timestamp} - INFO - Reading time values")
            # time_values = f["time"][:]

            with xr.open_dataset(h5_file, engine="h5netcdf") as ds:
                time_values = ds["time"].values

            time_length = len(time_values)
            print(f"{timestamp} - INFO - Found {time_length} time values")

            # Get lat/lon for all faces in this batch (only from first file)
            if file_idx == 0:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"{timestamp} - INFO - Reading lat/lon center values")

                if "lat_center" in f:
                    print(
                        f"{timestamp} - INFO - Reading lat_center[{start_face}:{end_face}]"
                    )
                    lat_values = f["lat_center"][start_face:end_face]
                    print(
                        f"{timestamp} - INFO - Read {len(lat_values)} lat_center values"
                    )
                    for i, face_id in enumerate(range(start_face, end_face)):
                        all_face_data[face_id]["lat"] = lat_values[i]

                if "lon_center" in f:
                    print(
                        f"{timestamp} - INFO - Reading lon_center[{start_face}:{end_face}]"
                    )
                    lon_values = f["lon_center"][start_face:end_face]
                    print(
                        f"{timestamp} - INFO - Read {len(lon_values)} lon_center values"
                    )
                    for i, face_id in enumerate(range(start_face, end_face)):
                        all_face_data[face_id]["lon"] = lon_values[i]

                # Early skip check: determine which faces already have output files
                if skip_existing:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    print(f"{timestamp} - INFO - Checking for existing output files...")
                    faces_to_skip = set()

                    for i, face_id in enumerate(range(start_face, end_face)):
                        lat = lat_values[i]
                        lon = lon_values[i]

                        # Compute expected output path using scalar-based functions
                        partition_dir = Path(
                            output_dir, get_partition_path_from_coords(lat, lon, config)
                        )
                        output_file = Path(
                            partition_dir,
                            get_partition_file_name_from_coords(
                                face_id, lat, lon, time_values, config, location
                            ),
                        )

                        # Check if file exists
                        if output_file.exists():
                            faces_to_skip.add(face_id)
                            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            print(
                                f"{timestamp} - INFO - Skipping face {face_id}: output file already exists ({output_file})"
                            )

                    # Update faces to process
                    if faces_to_skip:
                        # Remove skipped faces from all_face_data
                        for face_id in faces_to_skip:
                            del all_face_data[face_id]

                        faces_to_process = len(all_face_data)
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        print(
                            f"{timestamp} - INFO - Skipped {len(faces_to_skip)} existing files, {faces_to_process} faces remaining to process"
                        )

                        if faces_to_process == 0:
                            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            print(
                                f"{timestamp} - INFO - All files already exist, nothing to process"
                            )
                            return

            # Add time values to each face
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"{timestamp} - INFO - Adding time values to all faces")
            for face_id in all_face_data.keys():
                all_face_data[face_id]["time"].extend(time_values)

            # Efficiently read 2D datasets in batches
            for dataset_name in dataset_info["2d_datasets"]:
                if dataset_name in f and dataset_name not in [
                    "lat_center",
                    "lon_center",
                    "nv",
                ]:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    print(
                        f"{timestamp} - INFO - Reading {dataset_name}[:, {start_face}:{end_face}] (batch)"
                    )

                    # Read the entire batch of faces for all time points at once
                    batch_read_start = time.time()
                    data_batch = f[dataset_name][:, start_face:end_face]
                    batch_read_time = time.time() - batch_read_start

                    print(
                        f"{timestamp} - INFO - Read {dataset_name} batch with shape {data_batch.shape} in {batch_read_time:.2f} seconds"
                    )

                    # Distribute the data to each face
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    print(
                        f"{timestamp} - INFO - Distributing {dataset_name} data to individual faces"
                    )
                    distribute_start = time.time()

                    for i, face_id in enumerate(range(start_face, end_face)):
                        # Skip faces that were removed (when skip_existing is enabled)
                        if face_id not in all_face_data:
                            continue
                        # Extract this face's data (all time points)
                        face_data = data_batch[:, i]
                        all_face_data[face_id][dataset_name].extend(face_data)

                    distribute_time = time.time() - distribute_start
                    print(
                        f"{timestamp} - INFO - Distributed {dataset_name} data in {distribute_time:.2f} seconds"
                    )

            # Efficiently read 3D datasets - read ALL layers at once to reduce HDF5 operations
            # This is ~10x more efficient than reading layer-by-layer because HDF5 chunks
            # store all layers together, so reading one layer reads (and discards) all layers anyway
            for dataset_name, num_layers in dataset_info["3d_datasets"]:
                if dataset_name in f and dataset_name not in ["nv"]:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    print(
                        f"{timestamp} - INFO - Reading {dataset_name}[:, :, {start_face}:{end_face}] (all {num_layers} layers at once)"
                    )

                    # Read ALL layers at once - much more efficient than layer-by-layer
                    # Shape: (time, num_layers, num_faces_in_batch)
                    batch_read_start = time.time()
                    full_data_batch = f[dataset_name][:, :, start_face:end_face]
                    batch_read_time = time.time() - batch_read_start

                    print(
                        f"{timestamp} - INFO - Read {dataset_name} all layers batch with shape {full_data_batch.shape} in {batch_read_time:.2f} seconds"
                    )

                    # Now distribute each layer's data to faces (from memory, very fast)
                    distribute_start = time.time()
                    for layer_idx in range(num_layers):
                        col_name = f"{dataset_name}_layer_{layer_idx}"
                        # Slice the layer from memory (no disk I/O)
                        layer_data = full_data_batch[:, layer_idx, :]

                        for i, face_id in enumerate(range(start_face, end_face)):
                            # Skip faces that were removed (when skip_existing is enabled)
                            if face_id not in all_face_data:
                                continue
                            # Extract this face's data for this layer (all time points)
                            face_data = layer_data[:, i]
                            all_face_data[face_id][col_name].extend(face_data)

                    distribute_time = time.time() - distribute_start
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    print(
                        f"{timestamp} - INFO - Distributed {dataset_name} all {num_layers} layers to faces in {distribute_time:.2f} seconds"
                    )

        file_time = time.time() - file_start_time
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(
            f"{timestamp} - INFO - Completed reading file {file_idx + 1}/{len(h5_files)} in {file_time:.2f} seconds"
        )

    # =========================================================================
    # OPTIMIZED PARQUET WRITING WITH PRE-COMPUTED SCHEMA AND THREADED I/O
    # =========================================================================

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(
        f"{timestamp} - INFO - Writing {faces_to_process} parquet files using {PARQUET_WRITE_THREADS} threads"
    )
    writing_start = time.time()

    # -------------------------------------------------------------------------
    # Step 1: Build the column list and determine PyArrow types
    # -------------------------------------------------------------------------
    # Get a sample face to determine column structure
    sample_face_id = next(iter(all_face_data.keys()))
    sample_face_data = all_face_data[sample_face_id]
    time_length = len(sample_face_data["time"])

    # Define column order (time first, then coordinates, then data)
    coordinate_cols = [
        "lat",
        "lon",
        "element_corner_1_lat",
        "element_corner_1_lon",
        "element_corner_2_lat",
        "element_corner_2_lon",
        "element_corner_3_lat",
        "element_corner_3_lon",
    ]

    # Get data columns (everything except time and coordinates)
    data_cols = [
        k
        for k in sample_face_data.keys()
        if k
        not in ["time", "lat", "lon"] + [c for c in coordinate_cols if "corner" in c]
        and "nv" not in k
    ]

    # Full column order: time index (will be added), coordinates, data
    column_names = ["time"] + coordinate_cols + data_cols

    # Define PyArrow types for each column
    column_dtypes = {"time": pa.timestamp("ns")}
    for col in coordinate_cols:
        column_dtypes[col] = pa.float32()
    for col in data_cols:
        column_dtypes[col] = pa.float32()

    # -------------------------------------------------------------------------
    # Step 2: Build pre-computed schema with all metadata (ONCE)
    # -------------------------------------------------------------------------
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{timestamp} - INFO - Building pre-computed schema with metadata")

    template_schema = build_parquet_schema_with_metadata(
        column_names,
        column_dtypes,
        nc_metadata_for_parquet,
        parquet_col_to_nc_var_map,
    )

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{timestamp} - INFO - Schema built with {len(template_schema)} fields")

    # -------------------------------------------------------------------------
    # Step 3: Prepare all PyArrow tables and output paths
    # -------------------------------------------------------------------------
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{timestamp} - INFO - Preparing PyArrow tables for all faces")

    work_items = []
    prep_start = time.time()

    # Use list() to get keys upfront since we'll be modifying the dict
    face_ids_to_process = list(all_face_data.keys())

    for face_idx, face_id in enumerate(face_ids_to_process):
        # Pop face data from dict to free memory as we go
        # This prevents holding both source data AND tables in memory
        face_data = all_face_data.pop(face_id)

        # Get time values as numpy array
        time_values = np.array(face_data["time"], dtype="datetime64[ns]")

        # Build the data dict for PyArrow (no pandas!)
        pyarrow_data = {"time": time_values}

        # Add coordinate columns (repeat scalar values)
        for col in coordinate_cols:
            scalar_key = col
            if col == "lat":
                scalar_key = "lat"
            elif col == "lon":
                scalar_key = "lon"

            scalar_value = face_data.get(scalar_key)
            if scalar_value is not None:
                pyarrow_data[col] = np.repeat(np.float32(scalar_value), time_length)
            else:
                pyarrow_data[col] = np.repeat(np.float32(np.nan), time_length)

        # Add data columns
        for col in data_cols:
            if col in face_data:
                pyarrow_data[col] = np.array(face_data[col], dtype=np.float32)

        # Create PyArrow table with pre-computed schema
        table = pa.Table.from_pydict(pyarrow_data, schema=template_schema)

        # Compute output path (using scalar values, no DataFrame)
        lat = face_data["lat"]
        lon = face_data["lon"]

        partition_path = get_partition_path_from_coords(lat, lon, config)
        file_name = get_partition_file_name_from_coords(
            face_id, lat, lon, time_values, config, location
        )
        output_file = Path(output_dir) / partition_path / file_name

        work_items.append((face_id, table, output_file))

        # Progress and garbage collection every 1000 faces
        if (face_idx + 1) % 1000 == 0:
            # Force garbage collection to actually free the popped face data
            gc.collect()
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(
                f"{timestamp} - INFO - Prepared {face_idx + 1}/{faces_to_process} tables (memory freed via GC)"
            )

    # Final GC after all tables prepared
    gc.collect()

    prep_time = time.time() - prep_start
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(
        f"{timestamp} - INFO - Table preparation completed in {prep_time:.2f} seconds"
    )

    # -------------------------------------------------------------------------
    # Step 4: Write files using ThreadPoolExecutor with robust error handling
    # -------------------------------------------------------------------------
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(
        f"{timestamp} - INFO - Starting threaded file writes ({PARQUET_WRITE_THREADS} workers)"
    )

    successful_writes = []
    failed_writes = []
    write_start = time.time()

    with ThreadPoolExecutor(max_workers=PARQUET_WRITE_THREADS) as executor:
        # Submit all write jobs
        future_to_face = {
            executor.submit(write_parquet_file_safe, item): item[0]
            for item in work_items
        }

        # Process results as they complete
        completed_count = 0
        for future in as_completed(future_to_face):
            face_id, output_path, error = future.result()

            if error is None:
                successful_writes.append(face_id)
            else:
                failed_writes.append((face_id, output_path, error))

            completed_count += 1

            # Progress reporting every 100 files or at key milestones
            if completed_count % 100 == 0 or completed_count == faces_to_process:
                current_time = time.time()
                elapsed = current_time - write_start
                if completed_count > 0:
                    rate = completed_count / elapsed
                    remaining = (
                        (faces_to_process - completed_count) / rate if rate > 0 else 0
                    )
                else:
                    remaining = 0

                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(
                    f"{timestamp} - INFO - Written {completed_count}/{faces_to_process} files "
                    f"({rate:.1f} files/sec). Estimated remaining: {remaining:.1f}s"
                )

    write_time = time.time() - write_start
    total_writing_time = time.time() - writing_start

    # -------------------------------------------------------------------------
    # Step 5: Report results and handle any failures
    # -------------------------------------------------------------------------
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{timestamp} - INFO - File writing completed in {write_time:.2f} seconds")
    print(
        f"{timestamp} - INFO - Total write phase time: {total_writing_time:.2f} seconds"
    )
    print(
        f"{timestamp} - INFO - Successful writes: {len(successful_writes)}/{faces_to_process}"
    )

    if failed_writes:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"{timestamp} - ERROR - {len(failed_writes)} files FAILED to write:")
        for face_id, output_path, error in failed_writes[:10]:  # Show first 10
            print(f"  Face {face_id}: {error}")
        if len(failed_writes) > 10:
            print(f"  ... and {len(failed_writes) - 10} more failures")

        raise RuntimeError(
            f"{len(failed_writes)} parquet files failed to write. "
            f"First error: {failed_writes[0][2]}"
        )

    elapsed_time = time.time() - start_time
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(
        f"{timestamp} - INFO - Conversion complete! Total time: {elapsed_time:.2f} seconds"
    )


def partition_vap_into_parquet_dataset(
    config,
    location_key,
    batch_size=20000,
    batch_number=0,  # Batch number must start at zero
    skip_existing=False,
):
    """
    Process VAP data and convert to partitioned Parquet files using an optimized sequential approach.

    Parameters
    ----------
    config : dict
        Configuration dictionary
    location_key : str
        Key for location in the configuration
    batch_size : int, optional
        Number of faces to process in each batch (default: 20000)
    batch_number : int, optional
        Batch number to process (default: 0)
    skip_existing : bool, optional
        Skip processing faces whose output files already exist (default: False)
    """
    location = config["location_specification"][location_key]
    input_path = file_manager.get_vap_output_dir(config, location)
    output_path = file_manager.get_vap_partition_output_dir(
        config, location, use_temp_base_path=False
    )

    convert_h5_to_parquet_batched(
        input_path,
        output_path,
        config,
        location,
        batch_size=batch_size,
        batch_number=batch_number,
        skip_existing=skip_existing,
    )


if __name__ == "__main__":
    from config import config

    partition_vap_into_parquet_dataset(
        config, "cook_inlet", batch_size=2, batch_number=0
    )
