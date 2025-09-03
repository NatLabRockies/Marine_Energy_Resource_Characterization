from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import h5py

from config import config
from src.nc_manager import calculate_optimal_chunk_sizes


def create_hsds_tidal_dataset(
    input_path, output_path, timezone_offset, jurisdiction, include_vars=None
):
    """
    Convert monthly tidal fvcom b1 NC files to single H5 file following NREL HDF5 spec

    Parameters:
    - input_path: Directory containing monthly NC files
    - output_path: Output H5 file path
    - include_vars: List of variable patterns to include (None = all)
    """
    input_dir = Path(input_path)
    nc_files = sorted(list(input_dir.glob("*.nc")))

    if not nc_files:
        raise ValueError(f"No NC files found in {input_path}")

    print(f"Found {len(nc_files)} files to process")

    # Step 1: Analyze first file to understand structure
    with xr.open_dataset(nc_files[0]) as ds_first:
        n_faces = ds_first.dims["face"]
        n_sigma = ds_first.dims["sigma_layer"]

        # Get coordinate data (same for all files)
        lat_center = ds_first.lat_center.values
        lon_center = ds_first.lon_center.values

        # Get depth data from h_center variable
        water_depth = ds_first.h_center.values

        # Determine time steps per file
        time_steps_per_file = len(ds_first.time)

    print(
        f"Dataset structure: {n_faces} faces, {n_sigma} sigma layers, {time_steps_per_file} time steps per file"
    )

    # Step 2: Create metadata table (required by NREL spec)
    metadata = create_metadata_table(
        lat_center, lon_center, water_depth, timezone_offset, jurisdiction, n_faces
    )

    # Step 3: Process all files to create time series
    all_times = []
    all_data = {}

    for i, nc_file in enumerate(nc_files):
        print(f"Processing file {i + 1}/{len(nc_files)}: {nc_file.name}")

        with xr.open_dataset(nc_file) as ds:
            # Collect time information
            file_times = pd.to_datetime(ds.time.values)
            all_times.extend(file_times)

            # Process variables
            for var_name, var in ds.variables.items():
                if should_include_variable(var_name, include_vars):
                    process_variable(
                        var_name, var, all_data, i, n_sigma, time_steps_per_file
                    )

    # Step 4: Create time_index (required by NREL spec)
    time_index = create_time_index(all_times)

    # Step 5: Write H5 file
    write_h5_file(output_path, metadata, time_index, all_data, n_faces)

    print(f"Successfully created {output_path}")
    return output_path


def create_metadata_table(
    lat_center, lon_center, water_depth, timezone_offset, jurisdiction, n_faces
):
    """Create metadata table with exact structure: latitude, longitude, water_depth, timezone, jurisdiction"""
    # Create timezone array with user-specified offset
    timezone = np.full(
        n_faces, timezone_offset, dtype=np.int16
    )  # UTC offset in hours, int16

    # Create jurisdiction array with user-specified jurisdiction (use actual length)
    jurisdiction_bytes = jurisdiction.encode("utf-8")
    jurisdiction_dtype = f"S{len(jurisdiction_bytes)}"
    jurisdiction_array = np.full(n_faces, jurisdiction_bytes, dtype=jurisdiction_dtype)

    # Create structured array (HDF5 compound datatype) with EXACT types
    metadata_dtype = [
        ("latitude", "<f4"),  # little-endian float32
        ("longitude", "<f4"),  # little-endian float32
        ("water_depth", "<f4"),  # little-endian float32
        ("timezone", "<i2"),  # little-endian int16
        ("jurisdiction", jurisdiction_dtype),  # string with actual length
    ]

    metadata = np.empty(n_faces, dtype=metadata_dtype)
    metadata["latitude"] = lat_center.astype(np.float32)
    metadata["longitude"] = lon_center.astype(np.float32)
    metadata["water_depth"] = water_depth.astype(np.float32)  # Extract from h_center
    metadata["timezone"] = timezone
    metadata["jurisdiction"] = jurisdiction_array

    return metadata


def should_include_variable(var_name, include_vars):
    """Check if variable should be included based on include list"""
    if include_vars is None:
        return True

    # Check if any pattern in include_vars matches the variable name
    for pattern in include_vars:
        if pattern in var_name or var_name.startswith(pattern):
            return True
    return False


def process_variable(var_name, var, all_data, file_idx, n_sigma, time_steps_per_file):
    """Process a single variable and add to all_data dict"""
    dims = var.dims

    # Skip coordinate and connectivity variables
    skip_vars = [
        "lat_center",
        "lon_center",
        "lat_node",
        "lon_node",
        "nv",
        "face_node_index",
        "sigma_layer",
        "sigma_level",
        "time",
    ]
    if var_name in skip_vars:
        return

    if dims == ("face",):
        # 2D variable: just copy
        if var_name not in all_data:
            all_data[var_name] = []
        all_data[var_name].append(var.values)

    elif dims == ("sigma_layer", "face"):
        # 3D variable: create separate 2D variables for each sigma layer
        for sigma_idx in range(n_sigma):
            layer_var_name = f"{var_name}_sigma_{sigma_idx}"
            if layer_var_name not in all_data:
                all_data[layer_var_name] = []
            all_data[layer_var_name].append(var.values[sigma_idx, :])


def create_time_index(all_times):
    """Create time_index array in Unix timestamp format"""
    # Convert to Unix timestamps (seconds since 1970-01-01)
    time_index = np.array([t.timestamp() for t in all_times], dtype=np.float64)
    return time_index


def write_h5_file(output_path, metadata, time_index, all_data, n_faces):
    """Write data to H5 file following NREL spec"""
    n_times = len(time_index)

    with h5py.File(output_path, "w") as h5f:
        # Add version attribute
        h5f.attrs["version"] = "v1.0.0"

        # Create required datasets - rename to 'meta' to match specification
        h5f.create_dataset("meta", data=metadata)
        h5f.create_dataset("time_index", data=time_index)

        # Create data variables (2D: time × face)
        for var_name, var_data_list in all_data.items():
            # Stack time series data
            var_array = np.stack(var_data_list, axis=0)  # Shape: (n_times, n_faces)

            # Calculate optimal chunk sizes using shared chunking strategy
            # Verify dimensions match the array shape: (n_times, n_faces) -> ["time", "face"]
            array_shape = var_array.shape
            array_dims = ["time", "face"]  # Based on verified shape: (n_times, n_faces)

            chunk_size = calculate_optimal_chunk_sizes(
                shape=array_shape, dims=array_dims, dtype=var_array.dtype, config=config
            )

            h5f.create_dataset(
                var_name,
                data=var_array,
                chunks=chunk_size,
                compression="gzip",
                compression_opts=4,
            )

            print(f"Created variable {var_name} with shape {var_array.shape}")


if __name__ == "__main__":
    create_hsds_tidal_dataset(
        input_path="/projects/hindcastra/Tidal/datasets/high_resolution_tidal_hindcast/AK_cook_inlet/b1_vap",
        output_path="/projects/hindcastra/Tidal/datasets/high_resolution_tidal_hindcast/hsds/AK_cook_inlet.wpto_high_res_tidal.hsds.v0.3.0",
        timezone_offset=-9,
        jurisdiction="Alaska",
        include_vars=[
            "time",
            "latc",
            "lonc",
            "u",
            "v",
            "h_center",
            "vap_sea_water_speed",
            "vap_sea_water_to_direction",
            "vap_surface_elevation",
            "vap_sigma_depth",
            "vap_sea_floor_depth",
        ],
    )
