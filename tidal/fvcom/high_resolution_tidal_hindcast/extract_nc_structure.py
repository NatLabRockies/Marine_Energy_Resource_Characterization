import os

from datetime import datetime
from pathlib import Path

import xarray as xr
import json
import pandas as pd
import numpy as np

from config import config


def parse_nc_structure(nc_path, output_dir=None):
    """
    Parse an nc file using xarray and output its structure as JSON and parquet files.

    Parameters:
    -----------
    nc_path : str
        Path to the nc file
    output_dir : str, optional
        Directory to save the output files. If None, saves in the current directory.

    Returns:
    --------
    tuple
        Paths to the output JSON and parquet files
    """
    try:
        # Check if file exists
        if not os.path.exists(nc_path):
            raise FileNotFoundError(f"The file {nc_path} does not exist.")

        print(f"Opening dataset from {nc_path}")
        # Open the dataset
        ds = xr.open_dataset(nc_path)

        # Extract dimensions
        dimensions = {name: size for name, size in ds.dims.items()}
        print(f"Found {len(dimensions)} dimensions")

        # Extract coordinates
        coordinates = {}
        for name, var in ds.coords.items():
            coordinates[name] = {
                "shape": list(var.shape),
                "dtype": str(var.dtype),
                "dims": list(var.dims),
            }
        print(f"Found {len(coordinates)} coordinates")

        # Extract variables
        variables = {}
        for name, var in ds.variables.items():
            if name not in ds.coords:
                variables[name] = {
                    "shape": list(var.shape),
                    "dtype": str(var.dtype),
                    "dims": list(var.dims),
                    "attrs": {
                        k: _serialize_attr_value(v) for k, v in var.attrs.items()
                    },
                }
        print(f"Found {len(variables)} variables")

        # Extract global attributes
        attributes = {k: _serialize_attr_value(v) for k, v in ds.attrs.items()}
        print(f"Found {len(attributes)} global attributes")

        # Create the structure dictionary
        structure = {
            "dimensions": dimensions,
            "coordinates": coordinates,
            "variables": variables,
            "attributes": attributes,
        }

        # Get the datastream attribute for naming the output files
        datastream = ds.attrs.get("datastream", os.path.basename(nc_path).split(".")[0])
        print(f"Using datastream identifier: {datastream}")

        # Create a DataFrame for variables (for parquet output)
        variables_data = []

        for name, var in ds.variables.items():
            var_data = {
                "name": name,
                "is_coordinate": name in ds.coords,
                "dtype": str(var.dtype),
                "shape": str(list(var.shape)),
                "dimensions": str(list(var.dims)),
                "size": var.size,
                "ndim": var.ndim,
            }

            # Add all attributes as columns
            for k, v in var.attrs.items():
                attr_key = f"attr_{k}"
                var_data[attr_key] = _serialize_attr_value(v)

            variables_data.append(var_data)

        variables_df = pd.DataFrame(variables_data)

        # Close the dataset
        ds.close()

        # Set output directory
        if output_dir is None:
            output_dir = os.getcwd()
        os.makedirs(output_dir, exist_ok=True)

        # Create output filenames
        json_filename = f"{datastream}_structure.json"
        parquet_filename = f"{datastream}_variables.parquet"

        json_path = os.path.join(output_dir, json_filename)
        parquet_path = os.path.join(output_dir, parquet_filename)

        # Write to JSON file
        print(f"Writing JSON structure to {json_path}")
        with open(json_path, "w") as f:
            json.dump(structure, f, indent=2)

        # Write to parquet file
        print(f"Writing columnar data to {parquet_path}")
        variables_df.to_parquet(parquet_path, index=False)

        return json_path, parquet_path

    except Exception as e:
        print(f"Error processing {nc_path}: {str(e)}")


def _serialize_attr_value(v):
    """Helper function to serialize attribute values."""
    if isinstance(v, (bytes, bytearray)):
        return v.decode("utf-8", errors="replace")
    elif isinstance(v, datetime):
        return v.isoformat()
    elif isinstance(v, (np.ndarray, list, tuple)):
        return str(v)
    elif isinstance(v, (np.integer, np.floating)):
        return v.item()
    else:
        return v


if __name__ == "__main__":
    # Example usage
    locations = list(config["location_specification"].keys())
    location_keys = [
        config["location_specification"][loc]["output_name"] for loc in locations
    ]

    vap_directories = [
        config["dir"]["output"]["vap"].replace("<location>", key)
        for key in location_keys
    ]

    for dir in vap_directories:
        try:
            this_path = Path(dir)
            nc_files = sorted(list(this_path.glob("*.nc")))
            nc_file = nc_files[0]
            output_dir = "schema"  # Replace with your desired output directory
            parse_nc_structure(nc_file, output_dir)
        except:
            print("Error processing directory:", dir)
