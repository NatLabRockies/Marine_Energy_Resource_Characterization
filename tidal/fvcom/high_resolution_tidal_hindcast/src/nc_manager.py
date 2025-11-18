import math

import numpy as np
import xarray as xr


def nc_open(path, config, **kwargs):
    return xr.open_dataset(
        path, engine=config["dataset"]["xarray_netcdf4_engine"], **kwargs
    )


def verify_nv_dtype_in_memory(ds, config, context=""):
    """
    Verify that nv variable in dataset has the correct dtype.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to verify
    config : dict
        Configuration with standardized_variable_specification
    context : str, optional
        Context string for error message (e.g., "after reading", "before writing")

    Raises
    ------
    ValueError
        If nv dtype does not match expected dtype from config
    """
    if "nv" not in ds.data_vars:
        return

    expected_dtype = config["standardized_variable_specification"]["nv"]["dtype"]
    actual_dtype = str(ds.nv.dtype)

    context_str = f" {context}" if context else ""

    if actual_dtype != expected_dtype:
        raise ValueError(
            f"nv dtype{context_str}: {actual_dtype}, expected: {expected_dtype}"
        )


def verify_nv_dtype_on_disk(file_path, config, context=""):
    """
    Verify that nv variable in a NetCDF file has the correct dtype.

    Parameters
    ----------
    file_path : str or Path
        Path to NetCDF file to verify
    config : dict
        Configuration with standardized_variable_specification
    context : str, optional
        Context string for error message (e.g., "after writing")

    Raises
    ------
    ValueError
        If nv dtype does not match expected dtype from config
    """
    ds = nc_open(file_path, config)
    try:
        verify_nv_dtype_in_memory(ds, config, context)
    finally:
        ds.close()


def nc_write(ds, output_path, config, compression_strategy="none"):
    # Defensive fix: Ensure nv variable has correct dtype (int32) before writing
    # This prevents a dtype conversion bug that occurs in certain xarray/numpy version
    # combinations where nv gets converted from int32 to float32 during dataset operations
    if "nv" in ds.data_vars:
        expected_nv_dtype = config["standardized_variable_specification"]["nv"]["dtype"]
        current_nv_dtype = ds.nv.dtype

        if str(current_nv_dtype) != expected_nv_dtype:
            print(
                f"WARNING: nv variable has dtype {current_nv_dtype} but config "
                f"specifies {expected_nv_dtype}. Converting to {expected_nv_dtype}..."
            )
            ds["nv"] = ds.nv.astype(expected_nv_dtype)

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
    var = ds[var_name]

    # Use the shared chunking calculation
    chunk_sizes = calculate_optimal_chunk_sizes(
        shape=var.shape, dims=var.dims, dtype=var.dtype, config=config
    )

    this_encoding["chunksizes"] = chunk_sizes
    return this_encoding


def calculate_optimal_chunk_sizes(shape, dims, dtype, config):
    """
    Calculate optimal chunk sizes for any array (xarray or numpy) using the chunking strategy.

    Parameters:
    -----------
    shape : tuple
        Shape of the array (e.g., (n_times, n_faces))
    dims : list or tuple
        Dimension names corresponding to shape (e.g., ["time", "face"])
    dtype : numpy.dtype or str
        Data type of the array
    config : dict
        Configuration containing chunking specifications

    Returns:
    --------
    tuple
        Chunk sizes for each dimension (e.g., (100, 5000))
    """
    # Get chunking spec from config
    chunk_spec = config["dataset"]["encoding"]["chunk_spec"]
    target_chunk_size_mb = chunk_spec["target_size_mb"]
    target_chunk_size_bytes = target_chunk_size_mb * 1024 * 1024
    target_chunk_multiple = chunk_spec["multiple"]
    preferred_chunking_dimension = chunk_spec["preferred_dim"]

    # Convert dtype to numpy dtype if it's a string
    if isinstance(dtype, str):
        dtype = np.dtype(dtype)
    elif hasattr(dtype, "dtype"):  # xarray DataArray case
        dtype = dtype.dtype

    bytes_per_element = dtype.itemsize
    variable_bytes = math.prod(shape) * bytes_per_element

    # If the actual size is less than the target chunk size
    # no chunking is necessary and we just need to return the original shape
    if variable_bytes < target_chunk_size_bytes:
        return shape

    # Verify dims and shape have same length
    if len(shape) != len(dims):
        raise ValueError(
            f"Shape length ({len(shape)}) must match dims length ({len(dims)})"
        )

    dimension_sizes = {}
    for dim, size in zip(dims, shape):
        dimension_sizes[dim] = size

    # Find the chunking dimension
    if preferred_chunking_dimension in dims:
        chunking_dim = preferred_chunking_dimension
    else:
        # Fall back to largest dimension
        chunking_dim = None
        max_size = 0
        for this_dim, this_size in dimension_sizes.items():
            if this_size > max_size:
                max_size = this_size
                chunking_dim = this_dim

    # Create an array of sizes that are not the target chunking dim
    sizes = []
    for key, value in dimension_sizes.items():
        if key != chunking_dim:
            sizes.append(value)

    bytes_per_one_chunk_unit = math.prod(sizes) * bytes_per_element
    optimal_chunks_per_unit = target_chunk_size_bytes / bytes_per_one_chunk_unit

    # Calculate a chunk size for the target dimension that makes each chunk less than the target size
    # floor rounded to the multiple
    chunk_size = int(
        max(
            target_chunk_multiple,
            (optimal_chunks_per_unit // target_chunk_multiple) * target_chunk_multiple,
        )
    )

    # Chunking spec is a tuple of sizes for each chunk
    chunking_spec = []
    for dim, size in dimension_sizes.items():
        if dim == chunking_dim:
            chunking_spec.append(chunk_size)
        else:
            chunking_spec.append(size)

    return tuple(chunking_spec)


def define_numeric_encoding(ds, var_name, config, this_encoding):
    numeric_config = config["dataset"]["encoding"]["numeric"]
    float_type = numeric_config["default_float_type"]

    var = ds[var_name]

    # Set dtype for floating point variables
    if np.issubdtype(var.dtype, np.floating):
        this_encoding["dtype"] = float_type
    # Explicitly preserve integer dtypes to prevent unwanted type conversion
    elif np.issubdtype(var.dtype, np.integer):
        this_encoding["dtype"] = var.dtype

    return this_encoding
