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

# Global NaN replacement value
NAN_FILL_VALUE = -999.0


def replace_nans_with_fill_value(data, fill_value=NAN_FILL_VALUE):
    """
    Replace all NaN values in data with fill_value.

    Handles both floating point arrays and structured arrays (for meta compound dataset).

    Args:
        data: numpy array (can be regular array or structured array)
        fill_value: Value to replace NaNs with (default: -999.0)

    Returns:
        numpy array with NaNs replaced
    """
    # Make a copy to avoid modifying original
    data_filled = data.copy()

    # Handle structured arrays (meta compound dataset)
    if data.dtype.names is not None:
        for field_name in data.dtype.names:
            field_data = data_filled[field_name]
            # Only process numeric fields that can have NaNs
            if np.issubdtype(field_data.dtype, np.floating):
                nan_mask = np.isnan(field_data)
                nan_count = np.sum(nan_mask)
                if nan_count > 0:
                    field_data[nan_mask] = fill_value
                    print(f"    Replaced {nan_count} NaNs in field '{field_name}' with {fill_value}")
    # Handle regular arrays
    elif np.issubdtype(data.dtype, np.floating):
        nan_mask = np.isnan(data_filled)
        nan_count = np.sum(nan_mask)
        if nan_count > 0:
            data_filled[nan_mask] = fill_value
            print(f"    Replaced {nan_count} NaNs with {fill_value}")

    return data_filled


def add_fill_value_attr(dataset, fill_value=NAN_FILL_VALUE):
    """
    Add _FillValue attribute to dataset.

    Args:
        dataset: h5py Dataset object
        fill_value: Fill value to document in attributes
    """
    try:
        # Convert fill_value to appropriate dtype
        if np.issubdtype(dataset.dtype, np.floating):
            dataset.attrs['_FillValue'] = np.float32(fill_value)
            print(f"    Added _FillValue={fill_value} attribute")
        elif np.issubdtype(dataset.dtype, np.integer):
            dataset.attrs['_FillValue'] = int(fill_value)
            print(f"    Added _FillValue={int(fill_value)} attribute")
    except Exception as e:
        print(f"    Warning: Could not set _FillValue attribute: {e}")


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
        # Get variable information
        variable_info = {}
        meta_field_info = {}  # Face-only variables that will become meta fields
        total_time_steps = 0
        n_faces = None

        # First pass: collect basic info and identify face-only variables
        # Skip special datasets that are handled separately
        skip_datasets = {"time_index", "sigma_layer", "sigma_level"}

        print("\nCollecting variable info from first temporal file:")
        print("Validating NREL spec compliance (only 1D or 2D variables allowed)...")

        invalid_3d_vars = []

        for var_name in h5f.keys():
            if var_name not in skip_datasets:
                var_dataset = h5f[var_name]
                dims = var_dataset.dims
                shape = var_dataset.shape
                dtype = var_dataset.dtype

                print(f"  {var_name}: shape={shape}, dtype={dtype}")

                # NREL spec validation: only 1D or 2D variables allowed
                if len(shape) > 2:
                    invalid_3d_vars.append((var_name, shape))

                # Determine if this is a face-only variable for meta
                if len(shape) == 1:
                    # This is a 1D variable - could be face-only (for meta) or time-only
                    # We'll determine this by checking if size matches n_faces
                    if n_faces is None:
                        # First 1D variable - assume this is n_faces
                        # We need latitude/longitude to always exist
                        if var_name == "latitude" or var_name == "longitude":
                            n_faces = shape[0]
                            print(f"  Determined n_faces={n_faces} from {var_name}")

                variable_info[var_name] = {
                    "dims": [d.label for d in dims] if dims else None,
                    "shape_template": shape,
                    "dtype": dtype,
                    "attrs": dict(var_dataset.attrs),
                }

        # Validate we found n_faces
        if n_faces is None:
            raise ValueError("Could not determine n_faces - latitude or longitude variable not found")

        print(f"\nNumber of faces: {n_faces}")

        # Raise error if any 3D variables found
        if invalid_3d_vars:
            error_msg = (
                "\n" + "="*80 + "\n"
                "ERROR: NREL spec violation - 3D variables detected in H5 file!\n"
                "="*80 + "\n"
                "The NREL standardization spec requires all data variables to be 1D or 2D.\n"
                "3D variables (time, sigma, face) must be split into separate 2D variables.\n\n"
                "Found the following 3D variables:\n"
            )
            for var_name, shape in invalid_3d_vars:
                error_msg += f"  - {var_name}: shape={shape}\n"
            error_msg += (
                "\nTo fix this issue:\n"
                "1. Ensure convert_b1_vap_tidal_data_for_hsds.py is splitting 3D variables\n"
                "2. Re-run the conversion step to regenerate temporal H5 files\n"
                "3. Each 3D variable should be split into N separate 2D variables:\n"
                "   {var_name}_sigma_layer_1, {var_name}_sigma_layer_2, ..., {var_name}_sigma_layer_N\n"
                "="*80
            )
            raise ValueError(error_msg)

    # Second pass: calculate total time steps across all files
    print("\nCalculating total time steps...")
    for temporal_file in temporal_files:
        with h5py.File(temporal_file, "r", rdcc_nbytes=HDF5_READ_CACHE) as h5f:
            time_steps = len(h5f["time_index"])
            total_time_steps += time_steps

    print(f"Total time steps across all files: {total_time_steps}")

    # Third pass: separate face-only variables (for meta) from time-varying variables
    print("\nSeparating face-only variables (meta fields) from time-varying variables:")

    # Standard field ordering for meta compound dataset
    meta_field_order = [
        "latitude",
        "longitude",
        "water_depth",
        "timezone",
        "jurisdiction",
        "element_vertex_1_lat",
        "element_vertex_1_lon",
        "element_vertex_2_lat",
        "element_vertex_2_lon",
        "element_vertex_3_lat",
        "element_vertex_3_lon",
    ]

    for var_name, var_info in list(variable_info.items()):
        shape_template = var_info["shape_template"]

        # Identify face-only variables: 1D with size == n_faces
        if len(shape_template) == 1 and shape_template[0] == n_faces:
            # This is a face-only variable - goes into meta
            print(f"  {var_name}: shape={shape_template} → META FIELD")
            meta_field_info[var_name] = var_info
            # Remove from regular variable_info (will be in meta instead)
            del variable_info[var_name]
        elif len(shape_template) == 2 and shape_template[1] == n_faces:
            # Time-varying 2D variable (time, face)
            yearly_shape = (total_time_steps, n_faces)
            var_info["yearly_shape"] = yearly_shape
            var_info["is_static"] = False
            print(f"  {var_name}: shape={shape_template} → Time-varying (yearly: {yearly_shape})")
        else:
            # Other patterns - keep as-is (shouldn't happen in this dataset)
            var_info["yearly_shape"] = shape_template
            var_info["is_static"] = True
            print(f"  {var_name}: shape={shape_template} → Keeping as-is")

    # Sort meta fields by standard order, then alphabetically for any extras
    sorted_meta_fields = {}
    # First add fields in standard order
    for field_name in meta_field_order:
        if field_name in meta_field_info:
            sorted_meta_fields[field_name] = meta_field_info[field_name]

    # Then add any remaining fields alphabetically
    remaining_fields = sorted([f for f in meta_field_info.keys() if f not in meta_field_order])
    for field_name in remaining_fields:
        sorted_meta_fields[field_name] = meta_field_info[field_name]

    print(f"\nMeta will contain {len(sorted_meta_fields)} fields:")
    for field_name in sorted_meta_fields.keys():
        print(f"  - {field_name}")

    print(f"\nTime-varying variables: {len(variable_info)}")

    return {
        "n_faces": n_faces,
        "total_time_steps": total_time_steps,
        "n_temporal_files": len(temporal_files),
        "variable_info": variable_info,
        "meta_field_info": sorted_meta_fields,
    }


def assemble_h5_data_for_meta(template_file, meta_field_info, n_faces):
    """
    Read individual face-only datasets from template file and assemble into compound array.

    Args:
        template_file: Path to first temporal H5 file
        meta_field_info: Dict of field names to their info (dtype, attrs, etc.)
        n_faces: Number of faces

    Returns:
        numpy structured array ready for meta dataset
    """
    print("\nAssembling meta compound dataset:")

    # Build compound dtype from meta_field_info
    meta_dtype = []
    for field_name, field_info in meta_field_info.items():
        dtype = field_info["dtype"]
        meta_dtype.append((field_name, dtype))
        print(f"  {field_name}: {dtype}")

    # Create empty structured array
    meta_data = np.empty(n_faces, dtype=meta_dtype)

    # Read data from template file and populate structured array
    with h5py.File(template_file, "r", rdcc_nbytes=HDF5_READ_CACHE) as h5f:
        for field_name in meta_field_info.keys():
            if field_name not in h5f:
                raise ValueError(
                    f"Field '{field_name}' not found in template file. "
                    f"Available datasets: {list(h5f.keys())}"
                )

            field_data = h5f[field_name][:]
            meta_data[field_name] = field_data
            print(f"  Loaded {field_name}: shape={field_data.shape}, dtype={field_data.dtype}")

    # Replace NaNs in meta compound array
    print("\n  Replacing NaNs in meta fields:")
    meta_data = replace_nans_with_fill_value(meta_data)

    print(f"\nMeta compound array assembled: shape={meta_data.shape}, {len(meta_dtype)} fields")
    return meta_data


def assemble_attrs_for_meta(meta_field_info, existing_global_attrs):
    """
    Collect attributes from meta fields and format as global H5 attributes.

    Attributes are stored with pattern: meta:{field_name}:{attr_name} = attr_value

    Args:
        meta_field_info: Dict of field names to their info (including attrs)
        existing_global_attrs: Dict of existing global attributes to check for conflicts

    Returns:
        dict: Global attributes to add for meta fields

    Raises:
        ValueError: If attribute name conflicts with existing global attribute
    """
    print("\nAssembling attributes for meta fields:")

    meta_attrs = {}

    for field_name, field_info in meta_field_info.items():
        field_attrs = field_info.get("attrs", {})

        if field_attrs:
            print(f"  {field_name}: {len(field_attrs)} attributes")
            for attr_name, attr_value in field_attrs.items():
                # Format: meta:{field}:{attr}
                global_attr_name = f"meta:{field_name}:{attr_name}"

                # Check for conflicts with existing global attributes
                if global_attr_name in existing_global_attrs:
                    raise ValueError(
                        f"Attribute conflict: '{global_attr_name}' already exists in global attributes. "
                        f"Existing value: {existing_global_attrs[global_attr_name]}, "
                        f"New value: {attr_value}"
                    )

                meta_attrs[global_attr_name] = attr_value
                print(f"    {global_attr_name} = {attr_value}")
        else:
            print(f"  {field_name}: no attributes")

    print(f"Total meta attributes: {len(meta_attrs)}")
    return meta_attrs


def assemble_hsds_meta(template_file, meta_field_info, n_faces, existing_global_attrs):
    """
    Orchestrate meta compound dataset creation and attribute assembly.

    Args:
        template_file: Path to first temporal H5 file
        meta_field_info: Dict of field names to their info
        n_faces: Number of faces
        existing_global_attrs: Dict of existing global attributes

    Returns:
        tuple: (meta_data, meta_attrs)
            - meta_data: numpy structured array for meta dataset
            - meta_attrs: dict of global attributes for meta fields
    """
    print("\n" + "="*80)
    print("ASSEMBLING HSDS META DATASET")
    print("="*80)

    # Assemble the compound data array
    meta_data = assemble_h5_data_for_meta(template_file, meta_field_info, n_faces)

    # Assemble the attributes
    meta_attrs = assemble_attrs_for_meta(meta_field_info, existing_global_attrs)

    print("="*80)
    print("META ASSEMBLY COMPLETE")
    print("="*80 + "\n")

    return meta_data, meta_attrs


def create_yearly_file_structure(output_file, template_file, file_structure):
    """Create the structure of the yearly H5 file with meta compound dataset"""

    # First, assemble meta outside of file context
    existing_global_attrs = {}
    with h5py.File(template_file, "r", rdcc_nbytes=HDF5_READ_CACHE) as template_h5:
        existing_global_attrs = dict(template_h5.attrs)

    meta_data, meta_attrs = assemble_hsds_meta(
        template_file,
        file_structure["meta_field_info"],
        file_structure["n_faces"],
        existing_global_attrs,
    )

    with h5py.File(template_file, "r", rdcc_nbytes=HDF5_READ_CACHE) as template_h5:
        with h5py.File(output_file, "w", rdcc_nbytes=HDF5_WRITE_CACHE) as yearly_h5:
            # Copy global attributes from template
            for attr_name, attr_value in template_h5.attrs.items():
                yearly_h5.attrs[attr_name] = attr_value

            # Add meta field attributes as global attributes
            for attr_name, attr_value in meta_attrs.items():
                yearly_h5.attrs[attr_name] = attr_value

            # Create meta compound dataset
            meta_dataset = yearly_h5.create_dataset("meta", data=meta_data)
            # Add _FillValue as global attribute for meta (compound datasets can't have _FillValue directly)
            yearly_h5.attrs['meta:_FillValue'] = NAN_FILL_VALUE
            print(f"Created meta dataset: shape={meta_data.shape}, {len(meta_data.dtype.names)} fields")
            print(f"Added global attribute meta:_FillValue={NAN_FILL_VALUE}")

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

                # Add _FillValue attribute for numeric datasets
                add_fill_value_attr(dataset)


def stitch_data_into_yearly_file(output_file, monthly_files, file_structure):
    """
    Stitch data from monthly files into yearly file.

    Note: Face-only variables are already in meta compound dataset and are not stitched.
    Only time-varying variables are stitched across temporal chunks.
    """

    with h5py.File(output_file, "a", rdcc_nbytes=HDF5_WRITE_CACHE) as yearly_h5:
        print("\nStitching time-varying variables across temporal chunks...")
        print(f"Note: Face-only variables already assembled in meta dataset")
        print(f"Processing {len(file_structure['variable_info'])} time-varying variables\n")

        # Stitch time-varying variables across temporal chunks
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
                # Note: face-only variables (in meta) are not in variable_info and won't be processed here
                for var_name, var_info in file_structure["variable_info"].items():
                    monthly_data = monthly_h5[var_name][:]

                    # Replace NaNs with fill value
                    monthly_data = replace_nans_with_fill_value(monthly_data)

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

        # Time index is now string format: 'YYYY-MM-DD HH:MM:SS+00:00'
        # Decode bytes to strings and parse
        time_strings = [t.decode('utf-8') if isinstance(t, bytes) else str(t) for t in time_index]
        dt_timestamps = [pd.to_datetime(ts) for ts in time_strings]

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
