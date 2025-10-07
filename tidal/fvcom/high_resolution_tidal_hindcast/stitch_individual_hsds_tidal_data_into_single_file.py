#!/usr/bin/env python3
"""
Stitch together monthly HSDS files into a single yearly file.
Performs sanity checking on timestamps and ensures data continuity.
"""

import argparse
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import h5py

from config import config
from src.file_manager import get_hsds_temp_dir, get_hsds_final_file_path

# HDF5 cache settings for optimal performance
HDF5_READ_CACHE = config["hdf5_cache"]["read_cache_bytes"]
HDF5_WRITE_CACHE = config["hdf5_cache"]["stitch_write_cache_bytes"]


def stitch_single_b1_file_for_hsds(
    monthly_dir, output_file, location_name, perform_checks=True
):
    """
    Stitch together temporal HSDS files into single yearly file

    Parameters:
    - monthly_dir: Directory containing temporal H5 files
    - output_file: Output path for yearly H5 file
    - location_name: Location name for file pattern matching
    - perform_checks: Whether to perform sanity checks
    """
    monthly_dir = Path(monthly_dir)

    # Find all temporal files using consistent naming pattern (chunk_XXX)
    file_pattern = f"{location_name}_chunk_*_hsds.h5"
    metadata_pattern = f"{location_name}_chunk_*_metadata.json"

    # Find all temporal files and metadata
    temporal_files = sorted(list(monthly_dir.glob(file_pattern)))
    metadata_files = sorted(list(monthly_dir.glob(metadata_pattern)))

    if len(temporal_files) == 0:
        raise ValueError(
            f"No temporal HSDS files found matching pattern '{file_pattern}' in {monthly_dir}"
        )

    print(f"Found {len(temporal_files)} temporal files to stitch together")

    # Step 1: Load and validate metadata
    if perform_checks and len(metadata_files) > 0:
        print("Step 1: Validating temporal file metadata...")
        validate_temporal_metadata(metadata_files)

    # Step 2: Analyze structure from first file
    print("Step 2: Analyzing file structure...")
    file_structure = analyze_temporal_file_structure(temporal_files)

    # Step 3: Create yearly file structure
    print(f"Step 3: Creating yearly H5 file {output_file}")
    create_yearly_file_structure(output_file, temporal_files[0], file_structure)

    # Step 4: Stitch data from all temporal files
    print("Step 4: Stitching temporal data into yearly file...")
    stitch_data_into_yearly_file(output_file, temporal_files, file_structure)

    # Step 5: Final validation
    if perform_checks:
        print("Step 5: Performing final validation...")
        validate_yearly_file(output_file, temporal_files)

    print(f"Successfully created yearly HSDS file: {output_file}")
    return output_file


def validate_temporal_metadata(metadata_files):
    """Validate temporal metadata for continuity and completeness"""
    temporal_metadata = []
    for metadata_file in metadata_files:
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
            temporal_metadata.append(metadata)

    # Sort by chunk number
    temporal_metadata.sort(key=lambda x: x["chunk"])
    found_chunks = {m["chunk"] for m in temporal_metadata}
    expected_chunks = set(range(len(metadata_files)))

    # Check for sequential chunks
    if found_chunks != expected_chunks:
        raise ValueError(
            f"Missing or duplicate chunks. Expected {sorted(expected_chunks)}, found {sorted(found_chunks)}"
        )

    # All chunks must have same variables
    all_variables = [set(m["variables"]) for m in temporal_metadata]
    if len(set(frozenset(vars) for vars in all_variables)) > 1:
        raise ValueError("Variable lists differ between chunks")

    print(f"Metadata validation complete for {len(metadata_files)} chunks")


def analyze_temporal_file_structure(temporal_files):
    """Analyze structure of temporal files to determine yearly file dimensions"""
    print("Analyzing temporal file structure...")

    print(f"Reading structure from first file: {temporal_files[0].name}\n")

    with h5py.File(temporal_files[0], "r", rdcc_nbytes=HDF5_READ_CACHE) as h5f:
        # Get basic structure from first file
        n_faces = len(h5f["meta"])
        print(f"Number of faces: {n_faces}")

        # Get variable information
        variable_info = {}
        total_time_steps = 0

        # First pass: collect basic info
        # Skip special datasets that are handled separately
        skip_datasets = {"meta", "time_index", "sigma_layer", "sigma_level"}

        print("\nCollecting variable info from first temporal file:")
        for var_name in h5f.keys():
            if var_name not in skip_datasets:
                var_dataset = h5f[var_name]
                dims = var_dataset.dims
                shape = var_dataset.shape
                dtype = var_dataset.dtype

                print(f"  {var_name}: shape={shape}, dtype={dtype}")

                variable_info[var_name] = {
                    "dims": [d.label for d in dims] if dims else None,
                    "shape_template": shape,
                    "dtype": dtype,
                    "attrs": dict(var_dataset.attrs),
                }

    # Second pass: calculate total time steps across all files
    print("Calculating total time steps...")
    for temporal_file in temporal_files:
        with h5py.File(temporal_file, "r", rdcc_nbytes=HDF5_READ_CACHE) as h5f:
            time_steps = len(h5f["time_index"])
            total_time_steps += time_steps

    print(f"Total time steps across all files: {total_time_steps}")

    # Third pass: determine final shapes for yearly file
    # Define static variables that should NOT have time dimension added
    static_vars = {
        "lat_center",
        "lon_center",
        "lat_node",
        "lon_node",
        "nv",
        "face",
        "node",
        "sigma",
        "sigma_layer",
        "sigma_level",
        "h_center",
        "x_center",
        "y_center",
    }

    print("\nDetermining yearly shapes for each variable:")
    for var_name, var_info in variable_info.items():
        shape_template = var_info["shape_template"]
        print(f"  {var_name}:")
        print(f"    Template shape: {shape_template}")
        print(f"    Is in static_vars: {var_name in static_vars}")

        # Static variables keep their original shape (no time dimension)
        if var_name in static_vars:
            yearly_shape = shape_template
            print(f"    → Keeping static shape: {yearly_shape}")
        # For time-dependent variables, update time dimension
        elif len(shape_template) > 1:  # Multi-dimensional time-varying
            yearly_shape = (total_time_steps,) + shape_template[1:]
            print(f"    → Time-varying (multi-dim): {yearly_shape}")
        else:  # 1D time-varying
            yearly_shape = (total_time_steps,)
            print(f"    → Time-varying (1D): {yearly_shape}")

        var_info["yearly_shape"] = yearly_shape
        var_info["is_static"] = var_name in static_vars

    return {
        "n_faces": n_faces,
        "total_time_steps": total_time_steps,
        "n_temporal_files": len(temporal_files),
        "variable_info": variable_info,
    }


def create_yearly_file_structure(output_file, template_file, file_structure):
    """Create the structure of the yearly H5 file"""

    with h5py.File(template_file, "r", rdcc_nbytes=HDF5_READ_CACHE) as template_h5:
        with h5py.File(output_file, "w", rdcc_nbytes=HDF5_WRITE_CACHE) as yearly_h5:
            # Copy global attributes
            for attr_name, attr_value in template_h5.attrs.items():
                yearly_h5.attrs[attr_name] = attr_value

            # Copy metadata (same for all months)
            meta_data = template_h5["meta"][:]
            yearly_h5.create_dataset("meta", data=meta_data)

            # Copy sigma coordinates (same for all months)
            if "sigma_layer" in template_h5:
                sigma_layer_data = template_h5["sigma_layer"][:]
                yearly_h5.create_dataset("sigma_layer", data=sigma_layer_data)
                print(f"Copied sigma_layer: {sigma_layer_data.shape}")

            if "sigma_level" in template_h5:
                sigma_level_data = template_h5["sigma_level"][:]
                yearly_h5.create_dataset("sigma_level", data=sigma_level_data)
                print(f"Copied sigma_level: {sigma_level_data.shape}")

            # Create empty time_index with yearly size
            time_dtype = template_h5["time_index"].dtype
            yearly_h5.create_dataset(
                "time_index",
                shape=(file_structure["total_time_steps"],),
                dtype=time_dtype,
            )

            # Create empty datasets for all variables with yearly dimensions
            print("\nCreating yearly datasets:")
            for var_name, var_info in file_structure["variable_info"].items():
                yearly_shape = var_info["yearly_shape"]
                is_static = var_info.get("is_static", False)

                print(f"\n  Variable: {var_name}")
                print(f"    Shape: {yearly_shape}")
                print(f"    Dtype: {var_info['dtype']}")
                print(f"    Is static: {is_static}")

                # Create dataset with chunking
                if len(yearly_shape) > 1:
                    # Multi-dimensional: chunk by time and spatial dims
                    chunk_time = min(1000, yearly_shape[0])
                    chunk_spatial = (
                        min(10000, yearly_shape[1])
                        if len(yearly_shape) > 1
                        else yearly_shape[1]
                    )
                    chunks = (chunk_time, chunk_spatial)
                    if len(yearly_shape) > 2:
                        chunks = chunks + yearly_shape[2:]

                    print(f"    Chunks: {chunks}")

                    # Validate chunk size (rough estimate: chunks * dtype_size)
                    dtype_size = np.dtype(var_info["dtype"]).itemsize
                    chunk_elements = np.prod(chunks)
                    chunk_bytes = chunk_elements * dtype_size
                    chunk_gb = chunk_bytes / (1024**3)
                    print(f"    Estimated chunk size: {chunk_gb:.3f} GB ({chunk_bytes:,} bytes)")

                    if chunk_gb >= 4.0:
                        print(f"    WARNING: Chunk size exceeds 4GB limit!")
                else:
                    # 1D time series
                    chunks = (min(10000, yearly_shape[0]),)
                    print(f"    Chunks: {chunks}")

                print(f"    Creating dataset...")
                dataset = yearly_h5.create_dataset(
                    var_name,
                    shape=yearly_shape,
                    dtype=var_info["dtype"],
                    chunks=chunks,
                )
                print(f"    ✓ Dataset created successfully")

                # Copy attributes
                for attr_name, attr_value in var_info["attrs"].items():
                    try:
                        dataset.attrs[attr_name] = attr_value
                    except Exception as e:
                        print(
                            f"    Warning: Could not set attribute {attr_name}: {e}"
                        )


def stitch_data_into_yearly_file(output_file, monthly_files, file_structure):
    """Stitch data from monthly files into yearly file"""

    with h5py.File(output_file, "a", rdcc_nbytes=HDF5_WRITE_CACHE) as yearly_h5:
        # First, copy static variables from the first file (they're the same in all files)
        print("Copying static variables from first file...")
        with h5py.File(monthly_files[0], "r", rdcc_nbytes=HDF5_READ_CACHE) as first_h5:
            for var_name, var_info in file_structure["variable_info"].items():
                if var_info.get("is_static", False):
                    static_data = first_h5[var_name][:]
                    yearly_h5[var_name][:] = static_data
                    print(f"  Copied static {var_name}: {static_data.shape}")

        # Now stitch time-varying variables
        time_offset = 0
        for i, monthly_file in enumerate(monthly_files):
            print(
                f"Stitching temporal chunk {i + 1}/{len(monthly_files)}: {monthly_file.name}"
            )

            with h5py.File(
                monthly_file, "r", rdcc_nbytes=HDF5_READ_CACHE
            ) as monthly_h5:
                # Get time steps for this chunk
                monthly_time_steps = len(monthly_h5["time_index"])
                time_slice = slice(time_offset, time_offset + monthly_time_steps)

                # Copy time_index data
                yearly_h5["time_index"][time_slice] = monthly_h5["time_index"][:]

                # Copy time-varying variable data
                for var_name, var_info in file_structure["variable_info"].items():
                    if not var_info.get("is_static", False):
                        monthly_data = monthly_h5[var_name][:]
                        # Handle different dimensionalities
                        if len(monthly_data.shape) == 1:
                            yearly_h5[var_name][time_slice] = monthly_data
                        else:
                            yearly_h5[var_name][time_slice, ...] = monthly_data
                        print(f"  Copied {var_name}: {monthly_data.shape}")

                time_offset += monthly_time_steps

        print(f"Total time steps stitched: {time_offset}")


def validate_yearly_file(output_file, monthly_files):
    """Perform final validation on the yearly file"""
    with h5py.File(output_file, "r", rdcc_nbytes=HDF5_READ_CACHE) as yearly_h5:
        # Check time index continuity
        time_index = yearly_h5["time_index"][:]
        timestamps = [t[0] for t in time_index]  # Extract timestamp values

        # Convert to datetime for analysis
        dt_timestamps = [datetime.fromtimestamp(ts) for ts in timestamps]

        # Timestamps must be monotonic
        is_monotonic = all(
            dt_timestamps[i] <= dt_timestamps[i + 1]
            for i in range(len(dt_timestamps) - 1)
        )
        if not is_monotonic:
            raise ValueError("Timestamps are not monotonic")

        # Check data completeness
        total_expected_steps = 0
        for mf in monthly_files:
            with h5py.File(mf, "r", rdcc_nbytes=HDF5_READ_CACHE) as monthly_h5:
                total_expected_steps += len(monthly_h5["time_index"])

        actual_steps = len(yearly_h5["time_index"])
        if actual_steps != total_expected_steps:
            raise ValueError(
                f"Time step mismatch. Expected: {total_expected_steps}, Got: {actual_steps}"
            )

        print(f"Validation complete: {actual_steps} time steps")


def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(
        description="Stitch temporal HSDS files into yearly file"
    )
    parser.add_argument("--location", type=str, required=True, help="Location name")
    parser.add_argument(
        "--temp-dir",
        type=str,
        help="Directory with temporal H5 files (default from config)",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        help="Output yearly H5 file path (default from config)",
    )
    parser.add_argument(
        "--skip-checks", action="store_true", help="Skip validation checks"
    )

    args = parser.parse_args()

    # Get location config
    location_config = config["location_specification"][args.location]

    # Determine paths using file_manager
    if args.temp_dir:
        temp_dir = Path(args.temp_dir)
    else:
        temp_dir = get_hsds_temp_dir(config, location_config)

    if args.output_file:
        output_file = Path(args.output_file)
    else:
        output_file = get_hsds_final_file_path(config, location_config)

    output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"Stitching temporal files from {temp_dir}")
    print(f"Output: {output_file}")

    # Perform stitching
    stitch_single_b1_file_for_hsds(
        monthly_dir=temp_dir,
        output_file=output_file,
        location_name=args.location,
        perform_checks=not args.skip_checks,
    )

    print(f"Stitching complete! Final file: {output_file}")

    # Print final file info
    with h5py.File(output_file, "r", rdcc_nbytes=HDF5_READ_CACHE) as h5f:
        print(f"Final file size: {output_file.stat().st_size / (1024**3):.2f} GB")
        print(
            f"Variables: {[k for k in h5f.keys() if k not in ['meta', 'time_index']]}"
        )
        print(f"Time steps: {len(h5f['time_index'])}")


if __name__ == "__main__":
    main()
