# io module - Part of tidal_visualizer package

import os
from pathlib import Path
import xarray as xr
import pandas as pd


def find_dataset_file(base_dir, location_key, end_dir=""):
    """Find the netCDF dataset file for a given location."""
    path = Path(base_dir, location_key, end_dir)

    nc_files = sorted(list(path.rglob("*.nc")))

    if len(nc_files) == 0:
        raise ValueError(f"Found zero nc files in {str(path)}!")

    if len(nc_files) > 1:
        print(
            f"Found more than one nc file in {str(path)}, returning the first file found"
        )

    return nc_files[0]


def load_dataset(path):
    """Load an xarray dataset from a netCDF file."""
    return xr.open_dataset(path)


def get_variable_data(ds, variable_name, layer_index=0):
    """Extract variable data from a dataset, handling sigma layers if present."""
    if "sigma_layer" in ds[variable_name].dims:
        return ds[variable_name].isel(sigma_layer=layer_index).values
    else:
        return ds[variable_name].values


def export_hotspots_to_csv(hotspots, location_key, output_file):
    """Export hotspot data to a CSV file."""
    # Create DataFrame from hotspots
    df = pd.DataFrame(hotspots)

    # Expand the search_area_bounds column
    for bounds_key in ["min_lat", "max_lat", "min_lon", "max_lon"]:
        df[bounds_key] = df["search_area_bounds"].apply(lambda x: x[bounds_key])

    # Drop the original search_area_bounds column
    df = df.drop(columns=["search_area_bounds"])

    # Add location information
    df["location_key"] = location_key

    # Add timestamp
    from datetime import datetime

    df["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"Exported hotspot data to {output_file}")

    return output_file


def import_hotspots_from_csv(input_file):
    """Import hotspot data from a CSV file."""
    df = pd.read_csv(input_file)

    # Reconstruct search_area_bounds
    hotspots = []
    for _, row in df.iterrows():
        hotspot = row.to_dict()

        # Create the search_area_bounds dictionary
        hotspot["search_area_bounds"] = {
            "min_lat": hotspot.pop("min_lat"),
            "max_lat": hotspot.pop("max_lat"),
            "min_lon": hotspot.pop("min_lon"),
            "max_lon": hotspot.pop("max_lon"),
        }

        hotspots.append(hotspot)

    return hotspots
