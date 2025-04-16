import xarray as xr


def nc_open(path, config, **kwargs):
    return xr.open_dataset(
        path, engine=config["dataset"]["xarray_netcdf4_engine"], **kwargs
    )


def nc_write(ds, output_path, config, compression_strategy="none"):
    ds.to_netcdf(
        output_path,
        engine=config["dataset"]["xarray_netcdf4_engine"],
        encoding=define_compression_encoding(
            ds,
            base_encoding=config["dataset"]["encoding"],
            compression_strategy="none",
        ),
    )


def define_compression_encoding(
    this_ds, base_encoding=None, compression_strategy="standard", exclude_vars=None
):
    if exclude_vars is None:
        exclude_vars = []

    # Start with empty encoding dict if none provided
    encoding = {}

    # Copy base encoding if provided
    if base_encoding is not None:
        encoding = base_encoding.copy()

    # Determine compression level based on strategy
    complevel = 4  # Default standard compression

    if compression_strategy.lower() == "none":
        complevel = 0
    elif compression_strategy.lower() == "standard":
        complevel = 4
    elif compression_strategy.lower() == "archival":
        complevel = 9

    # Apply compression to all variables in the this_ds
    for var_name in this_ds.variables:
        # Skip variables in the exclude list
        if var_name in exclude_vars:
            continue

        # Initialize encoding for this variable if it doesn't exist
        if var_name not in encoding:
            encoding[var_name] = {}

        # Apply compression settings based on the strategy
        if complevel == 0:
            # No compression
            encoding[var_name]["zlib"] = False
        else:
            # Add compression with the determined level
            encoding[var_name]["zlib"] = True
            encoding[var_name]["complevel"] = complevel

    return encoding
