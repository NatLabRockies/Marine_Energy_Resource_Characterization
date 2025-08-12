import math

import numpy as np
import xarray as xr


def nc_open(path, config, **kwargs):
    return xr.open_dataset(
        path, engine=config["dataset"]["xarray_netcdf4_engine"], **kwargs
    )


def nc_write(ds, output_path, config, compression_strategy="none"):
    ds.to_netcdf(
        output_path,
        engine=config["dataset"]["xarray_netcdf4_engine"],
        encoding=define_ds_encoding(ds, config, compression_strategy),
    )


# Encoding configures the xarray engine specification for writing data
# This seems like it defined per variable. There is a lack of documentation regarding this step,
# But this step is important for defining how exactly the output netCDF (h5) files are structured
# The xarray defaults seem to be derived from the input nc files and this dataset has original data with
# questionable defaults.
# https://docs.xarray.dev/en/latest/generated/xarray.DataArray.encoding.html
# https://docs.unidata.ucar.edu/nug/current/netcdf_perf_chunking.html
def define_ds_encoding(ds, config, compression_strategy):
    encoding_config = config["dataset"]["encoding"]
    if "var" in encoding_config:
        base_encoding = encoding_config["var"].copy()
    else:
        base_encoding = None

    if base_encoding is None:
        result = {}
    else:
        result = base_encoding.copy()

    for var_name in ds.variables:
        this_encoding = {}

        this_encoding = define_compression_encoding(this_encoding, compression_strategy)

        this_encoding = define_chunk_size_encoding(ds, var_name, config, this_encoding)

        this_encoding = define_numeric_encoding(ds, var_name, config, this_encoding)

        result[var_name] = this_encoding

    return result


def define_compression_encoding(this_encoding, compression_strategy):
    # Determine compression level based on strategy
    complevel = 4  # Default standard compression

    if compression_strategy.lower() == "none":
        complevel = 0
    elif compression_strategy.lower() == "standard":
        complevel = 4
    elif compression_strategy.lower() == "archival":
        complevel = 9

    # Apply compression settings based on the strategy
    this_encoding["complevel"] = complevel

    if complevel == 0:
        # No compression
        this_encoding["zlib"] = False
    else:
        # Add compression with the determined level
        this_encoding["zlib"] = True

    return this_encoding


def define_chunk_size_encoding(ds, var_name, config, this_encoding):
    # Get chunking spec from config
    chunk_spec = config["dataset"]["encoding"]["chunk_spec"]
    target_chunk_size_mb = chunk_spec["target_size_mb"]
    target_chunk_size_bytes = target_chunk_size_mb * 1024 * 1024
    target_chunk_multiple = chunk_spec["multiple"]
    preferred_chunking_dimension = chunk_spec["preferred_dim"]

    var = ds[var_name]
    bytes_per_element = var.dtype.itemsize
    variable_bytes = var.size * bytes_per_element

    print(f"DEBUG: Processing variable '{var_name}'")
    print(f"DEBUG: Variable shape: {var.shape}, dims: {var.dims}")
    print(
        f"DEBUG: Target chunk size: {target_chunk_size_mb} MB ({target_chunk_size_bytes} bytes)"
    )
    print(
        f"DEBUG: Total variable size: {variable_bytes} bytes ({variable_bytes / 1024 / 1024:.2f} MB)"
    )

    # If the actual size is less than the target chunk size
    # no chunking is necessary and we just need to return the original shape
    if variable_bytes < target_chunk_size_bytes:
        print(
            f"DEBUG: Variable smaller than target chunk size, using original shape: {var.shape}"
        )
        this_encoding["chunksizes"] = var.shape
        return this_encoding

    dimension_sizes = {}
    for dim, size in zip(var.dims, var.shape):
        dimension_sizes[dim] = size

    print(f"DEBUG: Dimension sizes: {dimension_sizes}")
    print(f"DEBUG: Preferred chunking dimension: '{preferred_chunking_dimension}'")

    if preferred_chunking_dimension in var.dims:
        chunking_dim = preferred_chunking_dimension
        print(f"DEBUG: Using preferred chunking dimension: '{chunking_dim}'")
    else:
        chunking_dim = None
        max_size = 0
        for this_dim, this_size in dimension_sizes.items():
            if this_size > max_size:
                max_size = this_size
                chunking_dim = this_dim
        print(
            f"DEBUG: Preferred dim not found, using largest dimension: '{chunking_dim}' (size: {max_size})"
        )

    # Create an array of sizes that are not the target chunking dim
    sizes = []
    for key, value in dimension_sizes.items():
        if key != chunking_dim:
            sizes.append(value)

    print(f"DEBUG: Non-chunking dimension sizes: {sizes}")

    bytes_per_one_face = math.prod(sizes) * bytes_per_element
    optimal_chunks_per_face = target_chunk_size_bytes / bytes_per_one_face

    print(f"DEBUG: Bytes per face: {bytes_per_one_face}")
    print(f"DEBUG: Optimal chunks per face: {optimal_chunks_per_face}")

    # Calculate a chunk size for the target dimension that makes each chunk less than the target size
    # floor rounded to the multiple
    chunk_size = int(
        max(
            target_chunk_multiple,
            (optimal_chunks_per_face // target_chunk_multiple) * target_chunk_multiple,
        )
    )

    # CRITICAL FIX: Ensure chunk size doesn't exceed dimension size
    actual_dim_size = dimension_sizes[chunking_dim]
    chunk_size = min(chunk_size, actual_dim_size)

    print(f"DEBUG: Calculated chunk size: {chunk_size}")
    print(f"DEBUG: Actual dimension '{chunking_dim}' size: {actual_dim_size}")
    print(f"DEBUG: Final chunk size (capped): {chunk_size}")

    # Chunking spec is a tuple of sizes for each chunk
    chunking_spec = []
    for dim, size in dimension_sizes.items():
        if dim == chunking_dim:
            chunking_spec.append(chunk_size)
        else:
            chunking_spec.append(size)

    print(f"DEBUG: Final chunking spec: {tuple(chunking_spec)}")

    this_encoding["chunksizes"] = tuple(chunking_spec)
    return this_encoding


def define_numeric_encoding(ds, var_name, config, this_encoding):
    numeric_config = config["dataset"]["encoding"]["numeric"]
    float_type = numeric_config["default_float_type"]

    var = ds[var_name]

    # Set dtype for floating point variables
    if np.issubdtype(var.dtype, np.floating):
        this_encoding["dtype"] = float_type

    return this_encoding
