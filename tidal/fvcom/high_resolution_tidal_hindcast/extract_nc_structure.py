from datetime import datetime
from pathlib import Path
import json

import numpy as np
import pandas as pd
import xarray as xr

from config import config


def parse_nc_structure(nc_path, output_dir=None):
    nc_path = Path(nc_path)
    if not nc_path.exists():
        raise FileNotFoundError(f"The file {nc_path} does not exist.")

    print(f"Opening dataset from {nc_path}")
    ds = xr.open_dataset(nc_path)

    dimensions = {name: size for name, size in ds.sizes.items()}
    print(f"Found {len(dimensions)} dimensions")

    coordinates = {}
    for name, var in ds.coords.items():
        coordinates[name] = {
            "shape": list(var.shape),
            "dtype": str(var.dtype),
            "dims": list(var.dims),
        }
    print(f"Found {len(coordinates)} coordinates")

    variables = {}
    for name, var in ds.variables.items():
        if name not in ds.coords:
            variables[name] = {
                "shape": list(var.shape),
                "dtype": str(var.dtype),
                "dims": list(var.dims),
                "attrs": {k: _serialize_attr_value(v) for k, v in var.attrs.items()},
            }
    print(f"Found {len(variables)} variables")

    attributes = {k: _serialize_attr_value(v) for k, v in ds.attrs.items()}
    print(f"Found {len(attributes)} global attributes")

    structure = {
        "dimensions": dimensions,
        "coordinates": coordinates,
        "variables": variables,
        "attributes": attributes,
    }

    datastream = ds.attrs.get("datastream", nc_path.stem)
    print(f"Using datastream identifier: {datastream}")

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

        for k, v in var.attrs.items():
            attr_key = f"attr_{k}"
            var_data[attr_key] = _serialize_attr_value(v)

        variables_data.append(var_data)

    variables_df = pd.DataFrame(variables_data)

    ds.close()

    if output_dir is None:
        output_dir = Path.cwd()
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_filename = f"{datastream}_structure.json"
    parquet_filename = f"{datastream}_variables.parquet"

    json_path = output_dir / json_filename
    parquet_path = output_dir / parquet_filename

    print(f"Writing JSON structure to {json_path}")
    with open(json_path, "w") as f:
        json.dump(structure, f, indent=2)

    print(f"Writing columnar data to {parquet_path}")
    for col in variables_df.columns:
        if variables_df[col].dtype == "object":
            variables_df[col] = variables_df[col].astype(str)
    variables_df.to_parquet(parquet_path, index=False)

    return json_path, parquet_path


def _serialize_attr_value(v):
    if isinstance(v, (bytes, bytearray)):
        return v.decode("utf-8", errors="replace")
    elif isinstance(v, datetime):
        return v.isoformat()
    elif isinstance(v, (np.ndarray, list, tuple)):
        return str(v)
    elif isinstance(v, (np.integer, np.floating, np.bool_)):
        return v.item()
    else:
        return str(v) if v is not None else None


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
        this_path = Path(dir)
        nc_files = sorted(list(this_path.glob("*.nc")))
        nc_file = nc_files[0]
        output_dir = "schema"  # Replace with your desired output directory
        parse_nc_structure(nc_file, output_dir)
