from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from . import attrs_manager, file_manager, file_name_convention_manager


def verify_timestamps(nc_files, expected_timestamps, expected_delta_t_seconds):
    """Verify timestamp integrity across all files before processing."""

    total_timestamps = 0
    last_timestamp = None
    expected_diff = pd.Timedelta(seconds=expected_delta_t_seconds)

    for nc_file in nc_files:
        # Open dataset with minimal loading
        ds = xr.open_dataset(nc_file, decode_times=True)
        times = ds.time.values

        # Check count
        total_timestamps += len(times)

        # Check spacing within file
        time_diffs = pd.Series(times).diff()
        if not all(time_diffs[1:] == expected_diff):
            raise ValueError(f"Irregular timestamp spacing in {nc_file}")

        # Check continuity between files
        if last_timestamp is not None:
            gap = pd.Timestamp(times[0]) - last_timestamp
            if gap != expected_diff:
                raise ValueError(f"Gap between files: {nc_file}")

        last_timestamp = pd.Timestamp(times[-1])
        ds.close()

    if total_timestamps != expected_timestamps:
        raise ValueError(
            f"Expected {expected_timestamps} timestamps, found {total_timestamps}"
        )

    return True


def calculate_vap_average(config, location):
    """
    Calculate average values across VAP NC files using rolling computation.
    Only variables are averaged, preserving original dimensions and coordinates.
    Time dimension will use the first timestamp from the dataset.

    Args:
        config: Configuration dictionary
        location: Location dictionary containing site-specific parameters
    """
    location = config["location_specification"][location]
    vap_path = file_manager.get_vap_output_dir(config, location)
    vap_nc_files = sorted(list(vap_path.rglob("*.nc")))
    if len(vap_nc_files) < 12:
        raise ValueError(
            f"Expecting at least 12 files in {vap_path}, found {len(vap_nc_files)}: {vap_nc_files}"
        )

    # Typical year has 365 days
    seconds_per_year = 365 * 24 * 60 * 60
    expected_timestamps = int(seconds_per_year / location["expected_delta_t_seconds"])
    print("Verifying timestamps across all files...")
    verify_timestamps(
        vap_nc_files, expected_timestamps, location["expected_delta_t_seconds"]
    )

    # Initialize template using the first file
    print(f"Starting averaging of {len(vap_nc_files)} vap files...")
    running_sum = None
    total_times = 0
    source_files = []
    first_timestamp = None

    # Process each file
    for i, nc_file in enumerate(vap_nc_files):
        print(f"Processing File {i}: {nc_file}")
        ds = xr.open_dataset(nc_file)

        # Initialize running sum and save first timestamp if needed
        if running_sum is None:
            running_sum = ds.isel(time=0).copy(deep=True)
            first_timestamp = ds.time.isel(time=0).values
            # Only initialize variables that need averaging
            for var in running_sum.data_vars:
                if "time" in ds[var].dims:
                    running_sum[var].values.fill(0)

        # Update running sum for variables only
        for var in ds.data_vars:
            if "time" in ds[var].dims:
                running_sum[var] += ds[var].sum(dim="time")

        total_times += len(ds.time)
        source_files.append(str(nc_file))
        ds.close()

    # Calculate final average for variables only
    print("Computing final average...")
    for var in running_sum.data_vars:
        if var in running_sum.dims or var in running_sum.coords:
            continue  # Skip dimensions and coordinates
        if isinstance(running_sum[var].values, (np.ndarray, np.generic)):
            running_sum[var] = running_sum[var] / total_times

    # Restore the first timestamp
    running_sum["time"] = first_timestamp

    # Set up output dataset
    averaged_ds = running_sum

    # Generate output filename
    data_level_file_name = (
        file_name_convention_manager.generate_filename_for_data_level(
            averaged_ds,
            location["output_name"],
            config["dataset"]["name"],
            "b2",
            temporal="1_year_average",
        )
    )

    # Add standard attributes
    averaged_ds = attrs_manager.standardize_dataset_global_attrs(
        averaged_ds,
        config,
        location,
        "b2",
        source_files,
    )

    output_path = Path(
        file_manager.get_summary_vap_output_dir(config, location),
        f"001.{data_level_file_name}",
    )
    print(f"\tSaving to {output_path}...")
    averaged_ds.to_netcdf(output_path, encoding=config["dataset"]["encoding"])
    averaged_ds.close()
    return output_path
