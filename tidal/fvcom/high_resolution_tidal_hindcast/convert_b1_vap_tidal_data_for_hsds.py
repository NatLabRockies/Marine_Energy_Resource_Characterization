#!/usr/bin/env python3
"""
Convert single monthly tidal FVCOM b1 NC file to HSDS (HDF5) format following NREL spec.
This script processes one monthly file at a time for parallel processing.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import h5py

from config import config
from src.file_manager import get_vap_output_dir, get_hsds_temp_dir

# HDF5 cache settings for optimal performance
HDF5_WRITE_CACHE = config["hdf5_cache"]["write_cache_bytes"]
from src.nc_manager import calculate_optimal_chunk_sizes


def create_monthly_hsds_file(
    input_file, output_file, timezone_offset, jurisdiction_array, include_vars=None
):
    """
    Convert single monthly tidal FVCOM b1 NC file to HSDS format

    Parameters:
    - input_file: Path to single monthly NC file
    - output_file: Output H5 file path
    - timezone_offset: Array of UTC offsets in hours for each face
    - jurisdiction_array: Array of jurisdiction strings for each face (from VAP processing)
    - include_vars: List of variable patterns to include (None = all)
    """
    input_path = Path(input_file)
    if not input_path.exists():
        raise ValueError(f"Input file not found: {input_file}")

    print(f"Processing monthly file: {input_path.name}")

    # Step 1: Analyze file structure
    print("Step 1: Analyzing file structure...")
    file_info = analyze_file_structure([input_path], include_vars)

    n_faces = file_info["n_faces"]
    n_sigma = file_info["n_sigma"]
    time_steps = file_info["total_time_steps"]
    variable_info = file_info["variable_info"]
    global_attrs = file_info["global_attrs"]

    print(
        f"Monthly file structure: {n_faces} faces, {n_sigma} sigma layers, {time_steps} time steps"
    )

    # Step 2: Create metadata table
    print("Step 2: Creating metadata...")
    metadata = create_metadata_table(
        file_info["lat_center"],
        file_info["lon_center"],
        file_info["water_depth"],
        file_info["element_vertex_1_lat"],
        file_info["element_vertex_1_lon"],
        file_info["element_vertex_2_lat"],
        file_info["element_vertex_2_lon"],
        file_info["element_vertex_3_lat"],
        file_info["element_vertex_3_lon"],
        timezone_offset,
        jurisdiction_array,
        n_faces,
    )

    # Step 3: Stream monthly data to H5 file
    print(f"Step 3: Creating monthly H5 file {output_file}")
    stream_monthly_data_to_h5(
        input_path,
        output_file,
        metadata,
        file_info,
        variable_info,
        global_attrs,
        include_vars,
    )

    # Step 4: Return metadata for stitching validation
    time_info = {
        "start_time": file_info["all_times"][0],
        "end_time": file_info["all_times"][-1],
        "time_steps": time_steps,
        "variables": list(variable_info.keys()),
    }

    print(f"Successfully created monthly H5 file: {output_file}")
    return time_info


def analyze_file_structure(nc_files, include_vars=None):
    """Analyze structure of NC files (adapted from original)"""
    first_file = nc_files[0]

    with xr.open_dataset(first_file, decode_timedelta=False) as ds:
        # Get dimensions
        n_faces = len(ds.face)
        n_sigma = len(ds.sigma) if "sigma" in ds.dims else 0

        # Get coordinate data
        lat_center = ds.lat_center.values
        lon_center = ds.lon_center.values
        water_depth = ds.h_center.values  # Assuming h_center is water depth

        # Get node data for element vertexs
        lat_node = ds.lat_node.values
        lon_node = ds.lon_node.values
        nv = (
            ds.nv.values - 1
        )  # Convert from 1-based (Fortran) to 0-based (Python) indexing

        # Calculate element vertex coordinates using numpy indexing
        element_vertex_1_lat = lat_node[nv[0, :]]  # First vertex for all faces
        element_vertex_1_lon = lon_node[nv[0, :]]
        element_vertex_2_lat = lat_node[nv[1, :]]  # Second vertex for all faces
        element_vertex_2_lon = lon_node[nv[1, :]]
        element_vertex_3_lat = lat_node[nv[2, :]]  # Third vertex for all faces
        element_vertex_3_lon = lon_node[nv[2, :]]

        # Get global attributes
        global_attrs = dict(ds.attrs)

        # Collect all time coordinates
        all_times = []
        total_time_steps = 0

        for nc_file in nc_files:
            with xr.open_dataset(nc_file, decode_timedelta=False) as ds_temp:
                times = pd.to_datetime(ds_temp.time.values)
                all_times.extend(times)
                total_time_steps += len(times)

        # Sort times
        all_times = sorted(all_times)

        # Analyze variables
        variable_info = {}
        for var_name, var in ds.variables.items():
            if should_include_variable(var_name, include_vars):
                # Determine final shape for yearly dataset
                if "time" in var.dims:
                    shape = list(var.shape)
                    time_dim_idx = var.dims.index("time")
                    shape[time_dim_idx] = total_time_steps
                    final_shape = tuple(shape)
                else:
                    final_shape = var.shape

                # Convert dtype to h5py-compatible format
                dtype_str = np.dtype(var.dtype).str

                # Special handling for time variable - always store as string
                if var_name == "time":
                    # Time will be converted to string format (25 chars for YYYY-MM-DD HH:MM:SS+00:00)
                    dtype_str = "S25"
                    print("  Note: Converting time variable to string S25 for HDF5")
                # h5py cannot handle Unicode strings (<U)
                elif (
                    dtype_str.startswith("<U")
                    or dtype_str.startswith(">U")
                    or dtype_str.startswith("|U")
                ):
                    # Extract character length from Unicode dtype (e.g., '<U63' -> 63)
                    char_length = int(dtype_str.split("U")[1])
                    # Convert to byte string with same length
                    dtype_str = f"S{char_length}"
                    print(
                        f"  Note: Converting {var_name} from Unicode {var.dtype} to byte string S{char_length} for HDF5"
                    )
                elif "datetime64" in str(var.dtype) or "timedelta64" in str(var.dtype):
                    # Convert datetime/timedelta to string format (25 chars for YYYY-MM-DD HH:MM:SS+00:00)
                    dtype_str = "S25"
                    print(
                        f"  Note: Converting {var_name} from {var.dtype} to string S25 for HDF5"
                    )

                variable_info[var_name] = {
                    "shape": final_shape,
                    "dims": var.dims,
                    "dtype": dtype_str,
                    "attrs": dict(var.attrs),
                }

    return {
        "n_faces": n_faces,
        "n_sigma": n_sigma,
        "total_time_steps": total_time_steps,
        "lat_center": lat_center,
        "lon_center": lon_center,
        "water_depth": water_depth,
        "element_vertex_1_lat": element_vertex_1_lat,
        "element_vertex_1_lon": element_vertex_1_lon,
        "element_vertex_2_lat": element_vertex_2_lat,
        "element_vertex_2_lon": element_vertex_2_lon,
        "element_vertex_3_lat": element_vertex_3_lat,
        "element_vertex_3_lon": element_vertex_3_lon,
        "all_times": all_times,
        "variable_info": variable_info,
        "global_attrs": global_attrs,
    }


def create_metadata_table(
    lat_center,
    lon_center,
    water_depth,
    element_vertex_1_lat,
    element_vertex_1_lon,
    element_vertex_2_lat,
    element_vertex_2_lon,
    element_vertex_3_lat,
    element_vertex_3_lon,
    timezone_offset,
    jurisdiction_array,
    n_faces,
):
    """Create metadata table with per-face jurisdiction from VAP processing"""
    # timezone_offset is an array with individual values for each face
    timezone = timezone_offset.astype(np.int16)

    # Find the maximum length jurisdiction string to determine dtype
    max_jurisdiction_length = max(len(str(j)) for j in jurisdiction_array)
    jurisdiction_dtype = f"S{max_jurisdiction_length}"

    # Convert jurisdiction strings to bytes with consistent encoding
    jurisdiction_bytes_array = np.array(
        [str(j).encode("utf-8") for j in jurisdiction_array], dtype=jurisdiction_dtype
    )

    print(
        f"Jurisdiction strings: {len(np.unique(jurisdiction_array))} unique values, max length: {max_jurisdiction_length} characters"
    )

    metadata_dtype = [
        ("latitude", "<f4"),
        ("longitude", "<f4"),
        ("water_depth", "<f4"),
        ("timezone", "<i2"),
        ("jurisdiction", jurisdiction_dtype),
        ("element_vertex_1_lat", "<f4"),
        ("element_vertex_1_lon", "<f4"),
        ("element_vertex_2_lat", "<f4"),
        ("element_vertex_2_lon", "<f4"),
        ("element_vertex_3_lat", "<f4"),
        ("element_vertex_3_lon", "<f4"),
    ]

    metadata = np.empty(n_faces, dtype=metadata_dtype)
    metadata["latitude"] = lat_center.astype(np.float32)
    metadata["longitude"] = lon_center.astype(np.float32)
    metadata["water_depth"] = water_depth.astype(np.float32)
    metadata["timezone"] = timezone
    metadata["jurisdiction"] = jurisdiction_bytes_array
    metadata["element_vertex_1_lat"] = element_vertex_1_lat.astype(np.float32)
    metadata["element_vertex_1_lon"] = element_vertex_1_lon.astype(np.float32)
    metadata["element_vertex_2_lat"] = element_vertex_2_lat.astype(np.float32)
    metadata["element_vertex_2_lon"] = element_vertex_2_lon.astype(np.float32)
    metadata["element_vertex_3_lat"] = element_vertex_3_lat.astype(np.float32)
    metadata["element_vertex_3_lon"] = element_vertex_3_lon.astype(np.float32)

    return metadata


def stream_monthly_data_to_h5(
    nc_file, output_path, metadata, file_info, variable_info, global_attrs, include_vars
):
    """Stream single monthly file data to H5 file"""

    # Create time index for this monthly file
    with xr.open_dataset(nc_file, decode_timedelta=False) as ds:
        monthly_times = pd.to_datetime(ds.time.values)

    time_index = create_time_index(monthly_times)

    print(f"Creating monthly H5 file at {output_path}...")

    with h5py.File(output_path, "w", rdcc_nbytes=HDF5_WRITE_CACHE) as h5f:
        # Add global attributes
        if global_attrs:
            for attr_name, attr_value in global_attrs.items():
                try:
                    h5f.attrs[attr_name] = attr_value
                except Exception as e:
                    print(f"Warning: Could not set global attribute '{attr_name}': {e}")

        # Ensure version attribute
        if "version" not in h5f.attrs:
            h5f.attrs["version"] = "v1.0.0"

        # Create required datasets
        h5f.create_dataset("meta", data=metadata)
        h5f.create_dataset("time_index", data=time_index)

        # Process the monthly file
        print(f"Processing monthly file: {nc_file.name}")

        with xr.open_dataset(nc_file, decode_timedelta=False) as ds:
            time_steps_this_file = len(ds.time)

            # Create and populate datasets for this monthly file
            for var_name, var_info in variable_info.items():
                if var_name in ds.variables:
                    var = ds.variables[var_name]

                    if should_include_variable(var_name, include_vars):
                        print(
                            f"  Working on variable: {var_name} (dtype: {var_info['dtype']}, dims: {var_info['dims']})"
                        )

                        # For monthly files, adjust shape to match actual data
                        monthly_shape = list(var.shape)

                        # Use proven 6.4MB chunking strategy (yearly dimensions for consistency)
                        # This ensures same chunk pattern as final yearly file
                        yearly_shape = var_info[
                            "shape"
                        ]  # This is the yearly shape from analyze_file_structure

                        chunk_sizes = calculate_optimal_chunk_sizes(
                            shape=yearly_shape,  # Use yearly shape for consistent 6.4MB chunking
                            dims=var.dims,
                            dtype=var_info["dtype"],
                            config=config,
                        )

                        # Create and populate dataset
                        print(
                            f"    Creating dataset with shape {monthly_shape}, dtype {var_info['dtype']}, chunks {chunk_sizes}"
                        )
                        dataset = h5f.create_dataset(
                            var_name,
                            shape=monthly_shape,
                            dtype=var_info["dtype"],
                            chunks=chunk_sizes,
                        )

                        # Add attributes
                        for attr_name, attr_value in var_info["attrs"].items():
                            try:
                                dataset.attrs[attr_name] = attr_value
                            except Exception as e:
                                print(
                                    f"Warning: Could not set attribute '{attr_name}' for '{var_name}': {e}"
                                )

                        # Write data - handle string conversion if needed
                        print(f"    Writing data to {var_name}...")
                        data = var.values

                        # Special handling for time variable - convert to string format
                        if var_name == "time" and data.dtype.kind == "M":  # datetime64
                            # Convert datetime64 to pandas timestamps then to strings
                            times = pd.to_datetime(data)
                            time_strings = [datetime_to_hsds_string(t) for t in times]
                            # Determine max length
                            max_len = max(len(s) for s in time_strings)
                            # Convert to byte strings
                            data = np.array(
                                [s.encode("utf-8") for s in time_strings],
                                dtype=f"S{max_len}",
                            )
                            print(
                                f"      Converting datetime64 to string format S{max_len} for HDF5 storage"
                            )

                        # Convert Unicode strings to byte strings for h5py
                        elif data.dtype.kind == "U":  # Unicode string
                            # Convert Unicode array to byte string array
                            char_length = (
                                data.dtype.itemsize // 4
                            )  # Unicode uses 4 bytes per char
                            data = data.astype(f"S{char_length}")
                            print(
                                f"      Converting Unicode data to S{char_length} for HDF5 storage"
                            )

                        dataset[:] = data
                        print(f"    ✓ Completed variable: {var_name}")

    print(f"Successfully created monthly H5 file: {output_path}")


def datetime_to_hsds_string(dt):
    """
    Convert datetime to HSDS string format: 'YYYY-MM-DD HH:MM:SS+00:00'

    Args:
        dt: pandas Timestamp or datetime object

    Returns:
        str: formatted datetime string (25 characters)
    """
    # Ensure timezone-aware (assume UTC if not specified)
    if not hasattr(dt, "tz") or dt.tz is None:
        dt = pd.Timestamp(dt, tz="UTC")

    # Format: YYYY-MM-DD HH:MM:SS+00:00
    formatted = dt.strftime("%Y-%m-%d %H:%M:%S%z")

    # Insert colon in timezone offset to match format: +00:00 instead of +0000
    if len(formatted) == 24:  # YYYY-MM-DD HH:MM:SS+0000
        formatted = formatted[:-2] + ":" + formatted[-2:]

    return formatted


def create_time_index(times):
    """
    Create time index array with string timestamps

    Format: 'YYYY-MM-DD HH:MM:SS+00:00' (e.g., '1979-01-02 00:00:00+00:00')

    Args:
        times: list/array of pandas Timestamps

    Returns:
        numpy array of byte strings with dtype S<length>
    """
    # Convert all times to HSDS string format
    time_strings = [datetime_to_hsds_string(t) for t in times]

    # Verify all strings are the same length and determine dtype
    string_lengths = [len(s) for s in time_strings]
    max_length = max(string_lengths)
    min_length = min(string_lengths)

    if max_length != min_length:
        raise ValueError(
            f"Inconsistent datetime string lengths: min={min_length}, max={max_length}. "
            f"All timestamps must be the same length for HDF5 storage."
        )

    # Convert to bytes and create numpy array with appropriate dtype
    time_bytes = [s.encode("utf-8") for s in time_strings]
    time_index = np.array(time_bytes, dtype=f"S{max_length}")

    print(f"Created time_index with dtype S{max_length} for {len(times)} timestamps")

    return time_index


def should_include_variable(var_name, include_vars):
    """Check if variable should be included (same as original)"""
    if include_vars is None:
        return True

    # Skip coordinate variables by default
    skip_vars = [
        "lat_center",
        "lon_center",
        "lat_node",
        "lon_node",
        "nv",
        "face",
        "node",
        "sigma",
        "time",
        "h_center",
        "x_center",
        "y_center",
    ]

    if var_name in skip_vars:
        return False

    # Check against include patterns
    for pattern in include_vars:
        if pattern in var_name:
            return True

    return False


def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(
        description="Convert single temporal chunk of tidal data to HSDS format"
    )
    parser.add_argument(
        "--chunk",
        type=int,
        required=True,
        help="Chunk number (0-indexed: 0-11 for monthly, 0-72 for 5-day chunks)",
    )
    parser.add_argument(
        "--location", type=str, default="cook_inlet", help="Location name"
    )
    parser.add_argument(
        "--input-dir", type=str, help="Input directory (default from config)"
    )
    parser.add_argument(
        "--output-dir", type=str, help="Output directory (default from config)"
    )

    args = parser.parse_args()

    # Get location config
    location_config = config["location_specification"][args.location]

    # Determine input/output paths using file_manager
    if args.input_dir:
        input_dir = Path(args.input_dir)
    else:
        input_dir = get_vap_output_dir(config, location_config)

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = get_hsds_temp_dir(config, location_config)

    # Find all temporal chunk files using sorted listing
    all_files = sorted(list(input_dir.rglob("*.nc")))

    if not all_files:
        raise ValueError(f"No .nc files found in {input_dir}")

    # Validate chunk index
    if args.chunk < 0 or args.chunk >= len(all_files):
        raise ValueError(
            f"Chunk {args.chunk} out of range. Found {len(all_files)} files (valid range: 0-{len(all_files) - 1})"
        )

    # Select the file for this chunk
    input_file = all_files[args.chunk]
    output_file = output_dir / f"{args.location}_chunk_{args.chunk:03d}_hsds.h5"

    print(f"Converting {input_file} -> {output_file}")

    # Extract VAP data from input data (must be calculated in VAP processing)
    print("Reading VAP data from input file...")
    with xr.open_dataset(input_file) as ds:
        if "vap_utc_timezone_offset" not in ds.variables:
            raise ValueError(
                "Input file must contain 'vap_utc_timezone_offset' variable. Run VAP processing first."
            )
        if "vap_jurisdiction" not in ds.variables:
            raise ValueError(
                "Input file must contain 'vap_jurisdiction' variable. Run VAP processing first."
            )

        timezone_offset_array = ds["vap_utc_timezone_offset"].values
        jurisdiction_array = ds["vap_jurisdiction"].values

        # Validate jurisdiction data
        if len(jurisdiction_array) != len(ds.face):
            raise ValueError(
                f"Jurisdiction array length {len(jurisdiction_array)} does not match number of faces {len(ds.face)}"
            )

        # Check for any empty or missing jurisdiction strings
        empty_jurisdictions = np.sum(jurisdiction_array == "")
        if empty_jurisdictions > 0:
            raise ValueError(
                f"Found {empty_jurisdictions} faces with empty jurisdiction strings. All faces must have valid jurisdiction data."
            )

    # Process temporal file
    time_info = create_monthly_hsds_file(
        input_file=input_file,
        output_file=output_file,
        timezone_offset=timezone_offset_array,
        jurisdiction_array=jurisdiction_array,  # Now passing the full VAP jurisdiction array
    )

    # Save metadata for stitching validation
    metadata_file = output_dir / f"{args.location}_chunk_{args.chunk:03d}_metadata.json"
    import json

    metadata_content = {
        "chunk": args.chunk,
        "input_file": str(input_file),
        "output_file": str(output_file),
        "start_time": time_info["start_time"].isoformat(),
        "end_time": time_info["end_time"].isoformat(),
        "time_steps": time_info["time_steps"],
        "variables": time_info["variables"],
    }

    with open(metadata_file, "w") as f:
        json.dump(metadata_content, f, indent=2)

    print(f"Temporal conversion complete! Metadata saved to {metadata_file}")


if __name__ == "__main__":
    main()
