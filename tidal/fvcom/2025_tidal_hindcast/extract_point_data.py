from pathlib import Path

import xarray as xr
import pandas as pd

from config import config
from src import cli, file_manager


def extract_point_data(filename, this_point_index):
    """
    Extract all variables for a specific point from an FVCOM dataset

    Parameters:
    -----------
    filename : str or Path
        Path to the FVCOM netCDF file
    point_index : int
        Index of the point to extract

    Returns:
    --------
    pandas.DataFrame
        DataFrame containing all variables for the specified point
    """
    # Open the dataset
    ds = xr.open_dataset(filename)

    # Initialize an empty dictionary to store our data
    data_dict = {}

    # Get the time values
    time_values = ds.time.values

    # Process variables that vary with sigma
    sigma_vars = ["u", "v", "speed", "from_direction", "power_density", "depth"]
    for var in sigma_vars:
        if var in ds:
            # Extract data for all times and sigma levels at our point
            values = ds[var].isel(cell=this_point_index).values

            # Create columns for each sigma level
            for sigma_idx in range(ds.dims["sigma"]):
                if values.ndim == 2:  # time, sigma
                    data_dict[f"{var}_sigma_{sigma_idx}"] = values[:, sigma_idx]
                elif values.ndim == 3:  # time, sigma, other_dim
                    data_dict[f"{var}_sigma_{sigma_idx}"] = values[:, sigma_idx, 0]

    # Process variables that don't vary with sigma
    non_sigma_vars = [
        "zeta_center",
        "h_center",
        "seafloor_depth",
        "latitude",
        "longitude",
    ]
    for var in non_sigma_vars:
        if var in ds:
            values = ds[var].isel(cell=this_point_index).values
            if values.ndim > 1:
                data_dict[var] = values[:, 0]  # Take first element of extra dimensions
            else:
                data_dict[var] = values

    # Create DataFrame
    this_df = pd.DataFrame(data_dict, index=time_values)

    # Add metadata columns
    this_df["point_index"] = this_point_index

    # Close the dataset
    ds.close()

    return this_df


def process_all_files(directory, this_point_index, this_output_file):
    """
    Process all FVCOM files in a directory for a specific point

    Parameters:
    -----------
    directory : str or Path
        Directory containing FVCOM netCDF files
    point_index : int
        Index of the point to extract
    output_file : str or Path
        Path where the parquet file should be saved
    """
    # Convert to Path object
    directory = Path(directory)

    # Get all netCDF files
    nc_files = sorted(directory.glob("*.nc"))

    # Process each file and collect DataFrames
    dfs = []
    for file in nc_files:
        print(f"Processing {file}")
        this_df = extract_point_data(file, this_point_index)
        dfs.append(this_df)

    # Combine all DataFrames
    combined_df = pd.concat(dfs)

    # Sort by time index
    combined_df = combined_df.sort_index()

    # Save to parquet
    combined_df.to_parquet(this_output_file)
    print(f"Saved combined data to {this_output_file}")

    return combined_df


# Example usage
if __name__ == "__main__":
    args = cli.parse_args(config)

    location = config["location_specification"][args.location]
    location_output_name = location["output_name"]

    input_dir = file_manager.get_vap_output_dir(config, location)

    directory = input_dir
    # Cook Inlet East Forelands
    point_index = 125908
    output_dir = Path(
        f"/scratch/asimms/Tidal/{location_output_name}/b3_vap_by_point/"
    ).resolve()
    output_dir.mkdir(exist_ok=True)
    output_path = Path(
        output_dir, f"fvcom_{location_output_name}_point_{point_index}.parquet"
    )

    # Process all files
    df = process_all_files(directory, point_index, output_path)

    # Print some information about the result
    print("\nDataset Information:")
    print(f"Time range: {df.index.min()} to {df.index.max()}")
    print(f"Number of timestamps: {len(df)}")
    print(f"Columns: {', '.join(df.columns)}")
