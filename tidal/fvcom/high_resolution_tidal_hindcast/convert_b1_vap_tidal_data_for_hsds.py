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
    input_file, output_file, timezone_offset, jurisdiction_array
):
    """
    Convert single monthly tidal FVCOM b1 NC file to HSDS format

    Parameters:
    - input_file: Path to single monthly NC file
    - output_file: Output H5 file path
    - timezone_offset: Array of UTC offsets in hours for each face
    - jurisdiction_array: Array of jurisdiction strings for each face (from VAP processing)
    """
    input_path = Path(input_file)
    if not input_path.exists():
        raise ValueError(f"Input file not found: {input_file}")

    print(f"Processing monthly file: {input_path.name}")

    # Step 1: Analyze file structure (includes adding timezone and jurisdiction)
    print("Step 1: Analyzing file structure...")
    file_info = analyze_file_structure([input_path], timezone_offset, jurisdiction_array)

    n_faces = file_info["n_faces"]
    n_sigma = file_info["n_sigma"]
    time_steps = file_info["total_time_steps"]
    variable_info = file_info["variable_info"]
    global_attrs = file_info["global_attrs"]

    print(
        f"Monthly file structure: {n_faces} faces, {n_sigma} sigma layers, {time_steps} time steps"
    )

    # Step 2: Stream monthly data to H5 file (no meta creation)
    print(f"Step 2: Creating monthly H5 file {output_file}")
    stream_monthly_data_to_h5(
        input_path,
        output_file,
        file_info,
        variable_info,
        global_attrs,
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


def extract_and_verify_sigma_layers(ds):
    """
    Extract and verify sigma layer coordinates from FVCOM dataset.

    Sigma layers represent the vertical coordinate system in FVCOM.
    This function verifies that sigma_layer is uniform across all faces/nodes
    and calculates sigma_level (layer boundaries).

    Args:
        ds: xarray Dataset

    Returns:
        tuple: (sigma_layer, sigma_level) as 1D numpy arrays
    """
    if "sigma_layer" not in ds.variables:
        raise ValueError("Dataset does not contain 'sigma_layer' variable")

    sigma_layer_data = ds["sigma_layer"].values

    print("\nVerifying sigma_layer coordinate:")
    print(f"  Original shape: {sigma_layer_data.shape}")
    print(f"  Original dtype: {sigma_layer_data.dtype}")

    # Check if sigma_layer is 2D (sigma, face) or 1D (sigma)
    if sigma_layer_data.ndim == 2:
        # Verify all columns are identical
        first_column = sigma_layer_data[:, 0]
        all_identical = np.all(
            sigma_layer_data == first_column[:, np.newaxis], axis=1
        ).all()

        if not all_identical:
            # Check variance
            variance_per_layer = np.var(sigma_layer_data, axis=1)
            max_variance = np.max(variance_per_layer)
            print(
                f"  WARNING: sigma_layer varies across faces! Max variance: {max_variance}"
            )

            if max_variance > 1e-6:
                raise ValueError(
                    "sigma_layer is not uniform across faces. Expected identical values for all faces."
                )
            else:
                print("  Variance is negligible (<1e-6), treating as uniform")

        # Extract the uniform 1D array
        sigma_layer = first_column
        print(
            f"  Verified: sigma_layer is uniform across all {sigma_layer_data.shape[1]} faces"
        )

    elif sigma_layer_data.ndim == 1:
        sigma_layer = sigma_layer_data
        print("  sigma_layer is already 1D")
    else:
        raise ValueError(f"Unexpected sigma_layer dimensions: {sigma_layer_data.ndim}")

    # Round sigma_layer to clean up floating point errors
    # Assuming regular spacing around 0.05, 0.15, 0.25, etc.
    sigma_layer = np.round(sigma_layer, decimals=2)

    # Validate sigma_layer follows expected pattern: -0.05 - 0.1*n for n=0..9
    # This gives: [-0.05, -0.15, -0.25, -0.35, -0.45, -0.55, -0.65, -0.75, -0.85, -0.95]
    n_layers = len(sigma_layer)
    expected_sigma_layer = np.array([-0.05 - 0.1 * n for n in range(n_layers)])

    if not np.allclose(sigma_layer, expected_sigma_layer, atol=0.01):
        raise ValueError(
            f"sigma_layer does not follow expected pattern -0.05 - 0.1*n.\n"
            f"Expected: {expected_sigma_layer}\n"
            f"Got:      {sigma_layer}\n"
            f"Difference: {sigma_layer - expected_sigma_layer}"
        )

    # Replace with exact decimal values to avoid float32 precision errors
    sigma_layer = np.array([-0.05 - 0.1 * n for n in range(n_layers)], dtype=np.float32)

    print("  Validated: sigma_layer follows expected pattern")

    # Validate uniform spacing in sigma_layer
    sigma_layer_diffs = np.diff(sigma_layer)
    if not np.allclose(sigma_layer_diffs, sigma_layer_diffs[0], atol=0.001):
        raise ValueError(
            f"sigma_layer spacing is not uniform.\n"
            f"Differences: {sigma_layer_diffs}\n"
            f"Expected all differences to be approximately {sigma_layer_diffs[0]:.3f}"
        )
    print(f"  Validated: sigma_layer has uniform spacing of {sigma_layer_diffs[0]:.3f}")

    # Calculate sigma_level (layer boundaries)
    # sigma_level has n+1 elements where n is number of layers
    # Each boundary is midway between adjacent layer centers
    # Note: sigma ranges from 0 (surface) to -1 (bottom) in FVCOM convention

    # Construct sigma_level with exact decimal values
    # Formula: boundary[i] = -0.1 * i for i=0..10
    sigma_level = np.array([-0.1 * i for i in range(n_layers + 1)], dtype=np.float32)

    # Validate sigma_level ranges from 0 to -1
    if sigma_level[0] != 0.0:
        raise ValueError(f"sigma_level[0] must be 0.0 (surface), got {sigma_level[0]}")

    if not np.isclose(sigma_level[-1], -1.0, atol=0.01):
        raise ValueError(
            f"sigma_level[-1] must be -1.0 (bottom), got {sigma_level[-1]}"
        )

    # Ensure sigma_level is monotonically decreasing
    if not np.all(np.diff(sigma_level) < 0):
        raise ValueError(
            f"sigma_level must be monotonically decreasing from 0 to -1.\n"
            f"Got: {sigma_level}"
        )

    print(
        "  Validated: sigma_level ranges from 0.0 to -1.0 and is monotonically decreasing"
    )

    # Validate uniform spacing in sigma_level (excluding last boundary which is fixed at -1.0)
    sigma_level_diffs = np.diff(sigma_level[:-1])  # Exclude the last boundary
    if not np.allclose(sigma_level_diffs, sigma_level_diffs[0], atol=0.001):
        raise ValueError(
            f"sigma_level spacing is not uniform (excluding last boundary).\n"
            f"Differences: {sigma_level_diffs}\n"
            f"Expected all differences to be approximately {sigma_level_diffs[0]:.3f}"
        )
    print(
        f"  Validated: sigma_level has uniform spacing of {sigma_level_diffs[0]:.3f} (excluding last boundary)"
    )

    # Print full arrays for verification
    print(f"  sigma_layer ({n_layers} values): {sigma_layer}")
    print(f"  sigma_level ({n_layers + 1} values): {sigma_level}")

    return sigma_layer.astype(np.float32), sigma_level.astype(np.float32)


def analyze_file_structure(nc_files, timezone_offset=None, jurisdiction_array=None):
    """
    Analyze structure of NC files and prepare variable info.

    Args:
        nc_files: List of NetCDF file paths
        timezone_offset: Optional array of timezone offsets (for adding to variable_info)
        jurisdiction_array: Optional array of jurisdiction strings (for adding to variable_info)
    """
    first_file = nc_files[0]

    with xr.open_dataset(first_file, decode_timedelta=False) as ds:
        # Get dimensions
        n_faces = len(ds.face)
        n_sigma = len(ds.sigma_layer) if "sigma_layer" in ds.dims else 0

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

        # Extract and verify sigma layers (vertical coordinate)
        sigma_layer, sigma_level = extract_and_verify_sigma_layers(ds)

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

        # Analyze variables - apply NREL spec 3D to 2D splitting
        variable_info = {}

        # First, add the standard renamed variables as separate face-only datasets
        standard_face_vars = {
            "latitude": {
                "data": lat_center,
                "dtype": "<f4",
                "attrs": dict(ds.lat_center.attrs),
            },
            "longitude": {
                "data": lon_center,
                "dtype": "<f4",
                "attrs": dict(ds.lon_center.attrs),
            },
            "water_depth": {
                "data": water_depth,
                "dtype": "<f4",
                "attrs": dict(ds.h_center.attrs),
            },
        }

        # Add timezone and jurisdiction
        # Determine jurisdiction dtype
        max_jurisdiction_length = max(len(str(j)) for j in jurisdiction_array)
        jurisdiction_dtype = f"S{max_jurisdiction_length}"
        jurisdiction_bytes = np.array(
            [str(j).encode("utf-8") for j in jurisdiction_array], dtype=jurisdiction_dtype
        )

        standard_face_vars["timezone"] = {
            "data": timezone_offset.astype(np.int16),
            "dtype": "<i2",
            "attrs": dict(ds.vap_utc_timezone_offset.attrs),
        }

        standard_face_vars["jurisdiction"] = {
            "data": jurisdiction_bytes,
            "dtype": jurisdiction_dtype,
            "attrs": dict(ds.vap_jurisdiction.attrs),
        }

        # Add element vertex coordinates
        element_vertex_vars = {
            "element_vertex_1_lat": {"data": element_vertex_1_lat, "dtype": "<f4", "attrs": {}},
            "element_vertex_1_lon": {"data": element_vertex_1_lon, "dtype": "<f4", "attrs": {}},
            "element_vertex_2_lat": {"data": element_vertex_2_lat, "dtype": "<f4", "attrs": {}},
            "element_vertex_2_lon": {"data": element_vertex_2_lon, "dtype": "<f4", "attrs": {}},
            "element_vertex_3_lat": {"data": element_vertex_3_lat, "dtype": "<f4", "attrs": {}},
            "element_vertex_3_lon": {"data": element_vertex_3_lon, "dtype": "<f4", "attrs": {}},
        }

        # Add all standard and element vertex variables to variable_info
        for var_name, var_data in {**standard_face_vars, **element_vertex_vars}.items():
            variable_info[var_name] = {
                "shape": (n_faces,),
                "dims": ("face",),
                "dtype": var_data["dtype"],
                "attrs": var_data["attrs"],
                "original_variable": var_name,
                "sigma_layer_index": None,
                "static_data": var_data["data"],  # Store data for writing
            }

        # Now process all other variables from the dataset
        for var_name, var in ds.variables.items():
            if should_include_variable(var_name):
                dims = var.dims

                # Convert dtype to h5py-compatible format
                dtype_str = np.dtype(var.dtype).str

                # Special handling for time variable - always store as string
                if var_name == "time":
                    dtype_str = "S25"
                    print("  Note: Converting time variable to string S25 for HDF5")
                # h5py cannot handle Unicode strings (<U)
                elif (
                    dtype_str.startswith("<U")
                    or dtype_str.startswith(">U")
                    or dtype_str.startswith("|U")
                ):
                    char_length = int(dtype_str.split("U")[1])
                    dtype_str = f"S{char_length}"
                    print(
                        f"  Note: Converting {var_name} from Unicode {var.dtype} to byte string S{char_length} for HDF5"
                    )
                elif "datetime64" in str(var.dtype) or "timedelta64" in str(var.dtype):
                    dtype_str = "S25"
                    print(
                        f"  Note: Converting {var_name} from {var.dtype} to string S25 for HDF5"
                    )

                # Handle different dimension patterns following NREL spec
                if dims == ("time", "face"):
                    # 2D time series: direct use (time, face) -> (time, face)
                    final_shape = (total_time_steps, n_faces)
                    variable_info[var_name] = {
                        "shape": final_shape,
                        "dims": dims,
                        "dtype": dtype_str,
                        "attrs": dict(var.attrs),
                        "original_variable": var_name,
                        "sigma_layer_index": None,
                    }

                elif dims == ("time", "sigma", "face") or dims == (
                    "time",
                    "sigma_layer",
                    "face",
                ):
                    # 3D time series: split into sigma layers (1-indexed) - NREL spec
                    print(
                        f"  Note: Splitting 3D variable {var_name} into {n_sigma} sigma layers"
                    )
                    for sigma_idx in range(n_sigma):
                        layer_var_name = f"{var_name}_sigma_layer_{sigma_idx + 1}"
                        layer_attrs = dict(var.attrs)
                        layer_attrs["sigma_layer_index"] = sigma_idx + 1
                        layer_attrs["original_variable"] = var_name

                        variable_info[layer_var_name] = {
                            "shape": (total_time_steps, n_faces),
                            "dims": ("time", "face"),
                            "dtype": dtype_str,
                            "attrs": layer_attrs,
                            "original_variable": var_name,
                            "sigma_layer_index": sigma_idx,
                        }

                elif dims == ("face",):
                    # Static face-only variable
                    # Check if this is a standard renamed variable
                    output_var_name, is_renamed = get_renamed_variable_name(var_name)

                    # For vap_ variables (not standard renames), remove vap_ prefix
                    if not is_renamed and var_name.startswith("vap_"):
                        output_var_name = var_name[4:]  # Remove "vap_" prefix
                        print(f"  Note: Renaming {var_name} -> {output_var_name}")

                    # Special dtype handling for timezone and jurisdiction
                    if output_var_name == "timezone":
                        dtype_str = "<i2"  # int16 for timezone
                    elif output_var_name == "jurisdiction":
                        # Keep the byte string dtype from earlier conversion
                        pass

                    final_shape = (n_faces,)
                    variable_info[output_var_name] = {
                        "shape": final_shape,
                        "dims": dims,
                        "dtype": dtype_str,
                        "attrs": dict(var.attrs),
                        "original_variable": var_name,
                        "sigma_layer_index": None,
                    }

                else:
                    # Other dimension patterns - keep as-is
                    if "time" in dims:
                        shape = list(var.shape)
                        time_dim_idx = dims.index("time")
                        shape[time_dim_idx] = total_time_steps
                        final_shape = tuple(shape)
                    else:
                        final_shape = var.shape

                    variable_info[var_name] = {
                        "shape": final_shape,
                        "dims": dims,
                        "dtype": dtype_str,
                        "attrs": dict(var.attrs),
                        "original_variable": var_name,
                        "sigma_layer_index": None,
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
        "sigma_layer": sigma_layer,
        "sigma_level": sigma_level,
        "all_times": all_times,
        "variable_info": variable_info,
        "global_attrs": global_attrs,
    }


def stream_monthly_data_to_h5(
    nc_file, output_path, file_info, variable_info, global_attrs
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

        # Create time_index dataset
        h5f.create_dataset("time_index", data=time_index)

        # Create sigma coordinate datasets
        h5f.create_dataset("sigma_layer", data=file_info["sigma_layer"])
        h5f.create_dataset("sigma_level", data=file_info["sigma_level"])
        print(f"Created sigma_layer: {file_info['sigma_layer'].shape}")
        print(f"Created sigma_level: {file_info['sigma_level'].shape}")

        # Process the monthly file
        print(f"Processing monthly file: {nc_file.name}")

        with xr.open_dataset(nc_file, decode_timedelta=False) as ds:
            time_steps_this_file = len(ds.time)

            # Create and populate datasets for this monthly file
            for var_name, var_info in variable_info.items():
                # Check if this is a static face-only variable with pre-computed data
                if "static_data" in var_info:
                    # Static variable - write directly from stored data
                    print(
                        f"  Working on static variable: {var_name} (dtype: {var_info['dtype']}, dims: {var_info['dims']})"
                    )

                    data = var_info["static_data"]
                    monthly_shape = data.shape

                    # Create dataset
                    dataset = h5f.create_dataset(
                        var_name,
                        shape=monthly_shape,
                        dtype=var_info["dtype"],
                    )

                    # Add attributes
                    for attr_name, attr_value in var_info["attrs"].items():
                        try:
                            dataset.attrs[attr_name] = attr_value
                        except Exception as e:
                            print(
                                f"Warning: Could not set attribute '{attr_name}' for '{var_name}': {e}"
                            )

                    # Write data
                    dataset[:] = data
                    print(f"    ✓ Completed static variable: {var_name}")
                    continue

                # Get original variable name (for split sigma layers or renamed variables)
                original_var_name = var_info.get("original_variable", var_name)
                sigma_layer_index = var_info.get("sigma_layer_index", None)

                # Check if original variable exists in dataset
                if original_var_name in ds.variables:
                    original_var = ds.variables[original_var_name]

                    print(
                        f"  Working on variable: {var_name} (dtype: {var_info['dtype']}, dims: {var_info['dims']})"
                    )

                    # Determine monthly shape for this variable
                    if sigma_layer_index is not None:
                        # This is a split 3D variable - monthly shape is (time, face)
                        monthly_shape = (time_steps_this_file, len(ds.face))
                    else:
                        # Regular variable - use actual shape from dataset
                        monthly_shape = tuple(original_var.shape)

                    # Use proven 6.4MB chunking strategy (yearly dimensions for consistency)
                    yearly_shape = var_info["shape"]

                    chunk_sizes = calculate_optimal_chunk_sizes(
                        shape=yearly_shape,
                        dims=var_info["dims"],
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

                    # Write data - extract appropriate slice if 3D variable
                    print(f"    Writing data to {var_name}...")

                    if sigma_layer_index is not None:
                        # Extract specific sigma layer from 3D variable
                        data = original_var.values[
                            :, sigma_layer_index, :
                        ]  # Shape: (time, face)
                        print(
                            f"      Extracted sigma layer {sigma_layer_index + 1} from 3D variable"
                        )
                    else:
                        # Regular variable - use as-is
                        data = original_var.values

                    # Convert Unicode strings to byte strings for h5py
                    if data.dtype.kind == "U":  # Unicode string
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


def get_renamed_variable_name(var_name):
    """
    Get the renamed variable name for output H5 file.
    Renames standard NREL variables and removes vap_ prefix from static face variables.

    Returns:
        tuple: (output_name, is_renamed)
    """
    # Standard NREL variable renames
    rename_map = {
        "lat_center": "latitude",
        "lon_center": "longitude",
        "h_center": "water_depth",
        "vap_utc_timezone_offset": "timezone",
        "vap_jurisdiction": "jurisdiction",
    }

    if var_name in rename_map:
        return rename_map[var_name], True

    # For other face-only variables starting with vap_, remove the prefix
    # (This will be applied during variable processing based on dims)
    return var_name, False


def should_include_variable(var_name):
    """Check if variable should be included in output"""
    # Skip coordinate and node-based variables that are not written as data
    skip_vars = [
        "lat_center",  # Written as "latitude"
        "lon_center",  # Written as "longitude"
        "h_center",  # Written as "water_depth"
        "vap_utc_timezone_offset",  # Written as "timezone"
        "vap_jurisdiction",  # Written as "jurisdiction"
        "lat_node",
        "lon_node",
        "nv",
        "face",
        "node",
        "sigma",
        "sigma_layer",  # sigma_layer is written as separate coordinate dataset
        "time",  # time is handled separately via time_index
        "x_center",
        "y_center",
        "x",
        "y",
        "zeta",  # Node-based variable (time, node)
    ]

    return var_name not in skip_vars


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
