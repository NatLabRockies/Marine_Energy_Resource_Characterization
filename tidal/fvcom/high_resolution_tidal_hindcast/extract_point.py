import hashlib
import json
import os
import time

from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import xarray as xr
import h5py
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from config import config


from src.file_manager import get_vap_output_dir


def find_closest_faces(
    nc_file: str, target_lat: float, target_lon: float, n_closest: int = 10
) -> Tuple[List[int], List[float]]:
    """
    Find the n closest faces to a target lat/lon coordinate.

    Parameters:
    -----------
    nc_file : str
        Path to the first NC file to get coordinates from
    target_lat : float
        Target latitude
    target_lon : float
        Target longitude
    n_closest : int
        Number of closest faces to return

    Returns:
    --------
    Tuple[List[int], List[float]]
        Tuple of (face_indices, distances) sorted by distance
    """
    print(f"Finding {n_closest} closest faces to ({target_lat:.4f}, {target_lon:.4f})")

    with h5py.File(nc_file, "r") as f:
        # Get face center coordinates
        lat_center = f["lat_center"][:]
        lon_center = f["lon_center"][:]

        print(f"Loaded {len(lat_center)} face coordinates")

        # Calculate distances using simple euclidean (good for small areas)
        # For global data, consider using haversine distance
        distances = np.sqrt(
            (lat_center - target_lat) ** 2 + (lon_center - target_lon) ** 2
        )

        # Get indices of n closest faces
        closest_indices = sorted(list(np.argsort(distances)[:n_closest]))
        closest_distances = distances[closest_indices]

        print(f"Closest faces: {closest_indices}")
        print(f"Distances: {closest_distances.tolist()}")

        return closest_indices, closest_distances.tolist()


def create_subset_netcdf(
    input_files: List[Path], face_indices: List[int], output_file: str
) -> None:
    """
    Create a subset NetCDF file containing only the specified faces.

    Parameters:
    -----------
    input_files : List[Path]
        List of input NC files
    face_indices : List[int]
        Face indices to extract
    output_file : str
        Output NetCDF file path
    """
    print(f"Creating subset NetCDF with {len(face_indices)} faces")

    # Use xarray to create the subset
    datasets = []

    for i, nc_file in enumerate(input_files):
        print(f"Processing file {i+1}/{len(input_files)}: {nc_file.name}")

        with xr.open_dataset(nc_file) as ds:
            # Select only the specified faces for face-dimension variables
            face_vars = {}

            for var_name, var in ds.data_vars.items():
                if "face" in var.dims:
                    # Extract only the selected faces
                    face_vars[var_name] = var.isel(face=face_indices)
                else:
                    # Keep non-face variables as-is (like time)
                    face_vars[var_name] = var

            # Also handle coordinates
            coords = {}
            for coord_name, coord in ds.coords.items():
                if "face" in coord.dims:
                    coords[coord_name] = coord.isel(face=face_indices)
                else:
                    coords[coord_name] = coord

            # Create new dataset with subset
            subset_ds = xr.Dataset(face_vars, coords=coords, attrs=ds.attrs)

            subset_ds = subset_ds.drop_vars(["zeta"], errors="raise")

            datasets.append(subset_ds)

    # Concatenate along time dimension if multiple files
    if len(datasets) > 1:
        print("Concatenating datasets along time dimension")
        combined_ds = xr.concat(datasets, dim="time")
    else:
        combined_ds = datasets[0]

    # Save to NetCDF
    print(f"Saving subset NetCDF to {output_file}")
    combined_ds.to_netcdf(output_file)
    combined_ds.close()


def create_subset_netcdf_mfdataset(
    input_files: List[Path], face_indices: List[int], output_file: str
) -> None:
    """
    Create a subset NetCDF file using xarray open_mfdataset for more efficient processing.

    This version opens all files at once and leverages xarray's lazy loading and
    parallel processing capabilities.

    Parameters:
    -----------
    input_files : List[Path]
        List of input NC files
    face_indices : List[int]
        Face indices to extract
    output_file : str
        Output NetCDF file path
    """
    print(f"Creating subset NetCDF with {len(face_indices)} faces using open_mfdataset")
    print(f"Processing {len(input_files)} files simultaneously")

    # Open all files as a single dataset with time concatenation
    print("Opening multiple files with xarray...")
    with xr.open_mfdataset(
        input_files,
        concat_dim="time",
        combine="nested",
        engine="h5netcdf",
        # parallel=True,
        # chunks={"time": "auto"},  # Enable chunking for large datasets
        # decode_times=True,
    ) as mf_ds:
        print(f"Opened combined dataset with shape: {mf_ds.dims}")
        print(f"Time range: {mf_ds.time.min().values} to {mf_ds.time.max().values}")

        # Create subset by selecting only the specified faces
        print(f"Selecting {len(face_indices)} faces from dataset...")

        # Create the subset dataset
        print("Creating subset dataset...")
        subset_ds = mf_ds.isel(face=face_indices)

        print(f"Subset dataset shape: {subset_ds.sizes}")

        subset_ds.to_netcdf(output_file, engine="h5netcdf")

        print(f"Successfully created subset NetCDF: {output_file}")


def extract_faces_to_parquet(
    combined_nc_file_path: Path,
    output_dir: str,
    config: Dict,
    location: Dict,
) -> None:
    dataset = xr.open_dataset(combined_nc_file_path, engine="h5netcdf")

    # Common variables needed for all faces
    time_values = dataset.time.values
    time_dim_len = len(time_values)

    # Pre-fetch static data for all faces
    batch_data = {
        "lat_center": dataset.lat_center.values,
        "lon_center": dataset.lon_center.values,
        "nv": dataset["nv"].isel(time=0).values.T,
    }

    # Pre-fetch lat_node and lon_node if they exist
    if "lat_node" in dataset.variables and "lon_node" in dataset.variables:
        batch_data["lat_node"] = dataset["lat_node"].values
        batch_data["lon_node"] = dataset["lon_node"].values

    # Variables to skip in extraction
    vars_to_skip = ["nv", "h_center"]

    vars_to_include = list(dataset.variables.keys())
    for var_name in vars_to_include:
        if var_name in vars_to_skip:
            continue

        var = dataset[var_name]

        # Extract data based on variable dimensions
        if "sigma_layer" in var.dims and "face" in var.dims and "time" in var.dims:
            # 3D variables (time, sigma_layer, face)
            print(f"Extracting 4D variable {var_name} with dims {var.dims}")
            # selected_data = var.isel(face=face_indices)
            selected_data = var

            batch_data[var_name] = {}
            for layer_idx in range(len(dataset.sigma_layer)):
                layer_data = selected_data.isel(sigma_layer=layer_idx)

                # Ensure time dimension is first
                if layer_data.dims[0] == "time" and layer_data.dims[1] == "face":
                    data_array = layer_data.values
                else:
                    data_array = layer_data.transpose("time", "face").values

                batch_data[var_name][layer_idx] = data_array

        elif "face" in var.dims and "time" in var.dims:
            # 2D variables (time, face)
            print(f"Extracting 3D variable {var_name} with dims {var.dims}")
            # faces_data = var.isel(face=face_indices)
            faces_data = var

            # Ensure time dimension is first
            if "time" in faces_data.dims:
                time_dim_idx = faces_data.dims.index("time")
                if time_dim_idx != 0:
                    dim_order = list(faces_data.dims)
                    dim_order.remove("time")
                    dim_order.insert(0, "time")
                    faces_data = faces_data.transpose(*dim_order)

            batch_data[var_name] = faces_data.values

    # Create DataFrames for each face
    face_dataframes = {}
    face_indices = range(dataset.dims["face"])
    for i, face_idx in enumerate(face_indices):
        data_dict = {}

        # Add center coordinates
        data_dict["lat_center"] = [batch_data["lat_center"][i]] * time_dim_len
        data_dict["lon_center"] = [batch_data["lon_center"][i]] * time_dim_len

        # Process node vertex data
        nv_data = batch_data["nv"][i]
        node_indices = [int(idx - 1) if idx > 0 else int(idx) for idx in nv_data]

        # Add corner node data if available
        if "lat_node" in dataset.variables and "lon_node" in dataset.variables:
            for j, node_idx in enumerate(node_indices):
                corner_num = j + 1
                lat_node_val = float(batch_data["lat_node"][node_idx])
                lon_node_val = float(batch_data["lon_node"][node_idx])

                data_dict[f"element_corner_{corner_num}_lat"] = np.repeat(  # type: ignore
                    lat_node_val, time_dim_len
                )
                data_dict[f"element_corner_{corner_num}_lon"] = np.repeat(  # type: ignore
                    lon_node_val, time_dim_len
                )

        # Add variable data for this face
        for var_name in vars_to_include:
            if var_name in vars_to_skip or var_name not in batch_data:
                continue

            if (
                "sigma_layer" in dataset[var_name].dims
                and "face" in dataset[var_name].dims
            ):
                # Handle layered variables
                for layer_idx in range(len(dataset.sigma_layer)):
                    col_name = f"{var_name}_layer_{layer_idx}"
                    var_data = batch_data[var_name][layer_idx][:, i]

                    # Verify data length
                    if len(var_data) != time_dim_len:
                        print(
                            f"Warning: {col_name} for face {face_idx} has time dimension {len(var_data)} != {time_dim_len}"
                        )
                        continue

                    data_dict[col_name] = var_data

            elif "face" in dataset[var_name].dims and "time" in dataset[var_name].dims:
                # Handle 2D variables
                var_data = batch_data[var_name][:, i]

                # Verify data length
                if len(var_data) != time_dim_len:
                    print(
                        f"Warning: {var_name} for face {face_idx} has time dimension {len(var_data)} != {time_dim_len}"
                    )
                    continue

                data_dict[var_name] = var_data

        # Create DataFrame with time index
        df = pd.DataFrame(data_dict, index=time_values)
        df.index.name = "time"
        face_dataframes[face_idx] = df

    for this_df in face_dataframes.values():
        lat = f"{this_df['lat_center'].iloc[0]:.6f}"
        lon = f"{this_df['lon_center'].iloc[0]:.6f}"
        input_filename = Path(combined_nc_file_path).stem
        loc_name = config["location_specification"]["puget_sound"]["output_name"]
        temporal_resolution = config["location_specification"]["puget_sound"][
            "temporal_resolution"
        ]
        ds_name = config["dataset"]["name"]
        start_time = this_df.index.min().strftime("%Y%m%d.%H%M")

        # Create a unique filename based on lat/lon and face index
        output_filename = f"{loc_name}.{ds_name}.lat={lat}.lon={lon}.b4.{start_time}.{temporal_resolution}.1-year.parquet"

        output_file_path = Path(output_dir, output_filename)
        print("Saving Face DataFrame to parquet:", output_file_path)
        this_df.to_parquet(output_file_path, index=True)
        print("Save Successful")

    return face_dataframes


def extract_point_data(
    config: Dict[str, Any],
    location_key: str,
    target_lat: float,
    target_lon: float,
    output_path: str,
    n_closest: int = 10,
) -> None:
    """
    Extract data for points closest to target coordinates.

    Parameters:
    -----------
    config : dict
        Configuration dictionary
    location_key : str
        Key for location in the configuration
    target_lat : float
        Target latitude
    target_lon : float
        Target longitude
    output_path : str
        Directory to save output files
    n_closest : int
        Number of closest points to extract
    """
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{timestamp} - Starting point data extraction")
    print(f"Target: ({target_lat:.4f}, {target_lon:.4f})")
    print(f"Number of closest points: {n_closest}")
    print(f"Output path: {output_path}")

    # Get location config
    location = config["location_specification"][location_key]

    # This would need to be adapted based on your file_manager structure
    # For now, assuming input_dir is in the location config
    input_dir = get_vap_output_dir(config, location)

    # Find NC files
    nc_files = sorted(list(input_dir.glob("*.nc")))
    if not nc_files:
        nc_files = sorted(list(input_dir.glob("*.h5")))
    if not nc_files:
        raise ValueError(f"No NC/H5 files found in {input_dir}")

    print(f"Found {len(nc_files)} files")

    # Find closest faces
    closest_faces, distances = find_closest_faces(
        str(nc_files[0]), target_lat, target_lon, n_closest
    )

    # Create output directory
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create subset NetCDF
    subset_nc_path = output_dir / f"subset_{n_closest}_faces.nc"
    # create_subset_netcdf(nc_files, closest_faces, str(subset_nc_path))
    create_subset_netcdf_mfdataset(nc_files, closest_faces, str(subset_nc_path))

    # Create parquet files
    parquet_dir = output_dir / "parquet_files"
    extract_faces_to_parquet(subset_nc_path, str(parquet_dir), config, location)

    # Save metadata
    metadata = {
        "target_lat": target_lat,
        "target_lon": target_lon,
        "n_closest": n_closest,
        "face_indices": closest_faces,
        "distances": distances,
        "input_files": [str(f) for f in nc_files],
        "extraction_time": datetime.now().isoformat(),
    }

    metadata_file = output_dir / "extraction_metadata.json"
    import json

    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    elapsed_time = time.time() - start_time
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{timestamp} - Extraction complete! Total time: {elapsed_time:.2f} seconds")
    print(f"Outputs saved to: {output_path}")


# You'll need to implement or import this function based on your existing code
def get_dataset_info(h5_file):
    """
    Extract dataset information from HDF5 file.
    This should match your existing implementation.
    """
    dataset_info = {"2d_datasets": [], "3d_datasets": [], "total_faces": 0}

    with h5py.File(h5_file, "r") as f:
        # Get total faces from lat_center or similar
        if "lat_center" in f:
            dataset_info["total_faces"] = len(f["lat_center"])

        # Identify 2D and 3D datasets
        for key in f.keys():
            if key in f:
                shape = f[key].shape
                if len(shape) == 2:  # (time, face)
                    dataset_info["2d_datasets"].append(key)
                elif len(shape) == 3:  # (time, layer, face)
                    dataset_info["3d_datasets"].append((key, shape[1]))

    return dataset_info


def create_face_subset_files(
    input_files: List[Path],
    face_indices: List[int],
    output_dir: str,
    force_recreate: bool = False,
) -> List[Path]:
    """
    Extract specified faces from each input file and save as individual NetCDF files.

    Parameters:
    -----------
    input_files : List[Path]
        List of input NC files
    face_indices : List[int]
        Face indices to extract from each file
    output_dir : str
        Output directory for subset files
    force_recreate : bool
        Force recreation of existing files

    Returns:
    --------
    List[Path]
        List of created subset file paths
    """
    output_dir = Path(output_dir)
    subset_files_dir = output_dir / "face_subset_files"
    subset_files_dir.mkdir(parents=True, exist_ok=True)

    print(f"Extracting {len(face_indices)} faces from {len(input_files)} files")
    print(f"Face indices: {face_indices}")

    created_files = []

    for i, input_file in enumerate(input_files):
        # Create output filename based on input filename
        input_stem = input_file.stem
        subset_filename = f"{input_stem}_faces_{len(face_indices)}.nc"
        subset_file_path = subset_files_dir / subset_filename

        # Skip if already exists and not forcing recreation
        if subset_file_path.exists() and not force_recreate:
            print(f"Subset file already exists, skipping: {subset_file_path.name}")
            created_files.append(subset_file_path)
            continue

        print(f"Processing file {i+1}/{len(input_files)}: {input_file.name}")
        print(f"  Creating: {subset_filename}")

        try:
            # Open the input file
            with xr.open_dataset(input_file, engine="h5netcdf") as ds:
                print(f"  Original dataset shape: {ds.sizes}")

                subset_ds = ds.isel(face=face_indices)
                # Drop problematic variables if they exist
                subset_ds = subset_ds.drop_vars(["zeta"], errors="ignore")

                print(f"  Subset dataset shape: {subset_ds.sizes}")

                # Save subset file
                subset_ds.to_netcdf(subset_file_path, engine="h5netcdf")
                subset_ds.close()

                created_files.append(subset_file_path)
                print(f"  ✓ Saved: {subset_file_path.name}")

        except Exception as e:
            print(f"  ✗ Error processing {input_file.name}: {e}")
            # Remove partial file if it exists
            if subset_file_path.exists():
                subset_file_path.unlink()
            continue

    print(f"\nCompleted creating {len(created_files)} subset files")
    return created_files


def combine_subset_files(
    subset_files: List[Path], output_file: str, skip_if_exists: bool = True
) -> None:
    """
    Combine face subset files along time dimension into a single NetCDF file.

    Parameters:
    -----------
    subset_files : List[Path]
        List of face subset NetCDF files (one per time period)
    output_file : str
        Output combined NetCDF file path
    skip_if_exists : bool
        Skip if output file already exists
    """
    output_path = Path(output_file)

    if skip_if_exists and output_path.exists():
        print(f"Combined NetCDF already exists: {output_file}")
        return

    print(f"Combining {len(subset_files)} subset files into {output_file}")

    # Sort files to ensure proper time ordering
    subset_files = sorted(subset_files)

    try:
        # Use xarray's open_mfdataset for efficient concatenation
        print("Opening multiple subset files...")
        with xr.open_mfdataset(
            subset_files,
            concat_dim="time",
            combine="nested",
            engine="h5netcdf",
            # parallel=True,  # Uncomment if you want parallel processing
        ) as combined_ds:
            print(f"Combined dataset shape: {combined_ds.sizes}")
            print(
                f"Time range: {combined_ds.time.min().values} to {combined_ds.time.max().values}"
            )

            # Save combined file
            print(f"Saving combined NetCDF to {output_file}")
            combined_ds.to_netcdf(output_file, engine="h5netcdf")

    except Exception as e:
        print(f"Error combining files: {e}")
        # Try alternative approach with manual concatenation
        print("Trying manual concatenation approach...")

        datasets = []
        for subset_file in subset_files:
            print(f"Loading {subset_file.name}")
            ds = xr.open_dataset(subset_file, engine="h5netcdf")
            datasets.append(ds)

        # Combine along time dimension
        print("Concatenating datasets...")
        combined_ds = xr.concat(datasets, dim="time")

        # Save combined file
        print(f"Saving combined NetCDF to {output_file}")
        combined_ds.to_netcdf(output_file, engine="h5netcdf")

        # Clean up
        for ds in datasets:
            ds.close()
        combined_ds.close()

    print(f"✓ Combined NetCDF saved: {output_file}")


def convert_combined_to_parquet(
    combined_nc_file: str,
    face_indices: List[int],
    output_dir: str,
    skip_existing: bool = True,
) -> None:
    """
    Convert combined NetCDF to individual parquet files for each face.

    Parameters:
    -----------
    combined_nc_file : str
        Path to combined NetCDF file
    face_indices : List[int]
        Original face indices (for naming)
    output_dir : str
        Output directory for parquet files
    skip_existing : bool
        Skip conversion if parquet files already exist
    """
    parquet_dir = Path(output_dir) / "parquet_files"
    parquet_dir.mkdir(parents=True, exist_ok=True)

    print("Converting combined NetCDF to parquet files")
    print(f"Input: {combined_nc_file}")
    print(f"Output directory: {parquet_dir}")

    # Open the combined NetCDF file
    with xr.open_dataset(combined_nc_file, engine="h5netcdf") as ds:
        print(f"Dataset shape: {ds.sizes}")

        # Convert each face to a separate parquet file
        for i, original_face_id in enumerate(face_indices):
            output_parquet = parquet_dir / f"face_{original_face_id:06d}.parquet"

            if skip_existing and output_parquet.exists():
                print(
                    f"Parquet file already exists for face {original_face_id}, skipping..."
                )
                continue

            print(f"Converting face {original_face_id} (index {i})")

            try:
                # Select this specific face (using index i, not original_face_id)
                face_ds = ds.isel(face=i)

                # Convert to DataFrame
                df = face_ds.to_dataframe()

                # Reset index to make time a column if it's in the index
                if "time" in df.index.names:
                    df = df.reset_index()

                # Ensure time is datetime
                if "time" in df.columns:
                    if not pd.api.types.is_datetime64_any_dtype(df["time"]):
                        df["time"] = pd.to_datetime(df["time"], unit="s", origin="unix")

                    # Set time as index and sort
                    df = df.set_index("time").sort_index()

                # Save as parquet
                table = pa.Table.from_pandas(df)
                pq.write_table(table, output_parquet)

                print(f"  ✓ Saved {output_parquet.name}")

            except Exception as e:
                print(f"  ✗ Error converting face {original_face_id}: {e}")
                continue

    print("✓ Parquet conversion complete")


def extract_point_data_incremental(
    config: Dict[str, Any],
    location_key: str,
    target_lat: float,
    target_lon: float,
    output_path: str,
    n_closest: int = 10,
    force_recreate: bool = False,
) -> None:
    """
    Extract data for points closest to target coordinates using incremental processing.

    Process:
    1. Extract specified faces from each input file -> subset files
    2. Combine subset files along time dimension -> single NetCDF
    3. Convert to individual parquet files for each face
    """
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{timestamp} - Starting incremental point data extraction")
    print(f"Target: ({target_lat:.4f}, {target_lon:.4f})")
    print(f"Number of closest points: {n_closest}")
    print(f"Output path: {output_path}")

    # Get location config
    location = config["location_specification"][location_key]

    # Get input files (adapt based on your file_manager structure)
    from src.file_manager import get_vap_output_dir

    input_dir = get_vap_output_dir(config, location)

    # Find NC files
    nc_files = sorted(list(input_dir.glob("*.nc")))
    if not nc_files:
        nc_files = sorted(list(input_dir.glob("*.h5")))
    if not nc_files:
        raise ValueError(f"No NC/H5 files found in {input_dir}")

    print(f"Found {len(nc_files)} input files")

    # Find closest faces
    closest_faces, distances = find_closest_faces(
        str(nc_files[0]), target_lat, target_lon, n_closest
    )

    # Create output directory
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Create face subset files (extract faces from each input file)
    print("\n" + "=" * 60)
    print("STEP 1: Creating face subset files from each input file")
    print("=" * 60)
    subset_files = create_face_subset_files(
        nc_files, closest_faces, str(output_dir), force_recreate
    )

    if not subset_files:
        raise ValueError("No subset files were created!")

    # Step 2: Combine subset files into single NetCDF along time dimension
    print("\n" + "=" * 60)
    print("STEP 2: Combining subset files along time dimension")
    print("=" * 60)
    combined_nc_path = output_dir / f"combined_{n_closest}_faces.nc"
    combine_subset_files(
        subset_files, str(combined_nc_path), skip_if_exists=not force_recreate
    )

    # Step 3: Convert to parquet files
    print("\n" + "=" * 60)
    print("STEP 3: Converting to individual parquet files")
    print("=" * 60)
    # convert_combined_to_parquet(
    #     str(combined_nc_path),
    #     closest_faces,
    #     str(output_dir),
    #     skip_existing=not force_recreate,
    # )
    extract_faces_to_parquet(
        combined_nc_path,
        output_dir,
        config,
        location,
    )

    # Save final metadata
    metadata = {
        "target_lat": target_lat,
        "target_lon": target_lon,
        "n_closest": n_closest,
        "face_indices": closest_faces,
        "distances": distances,
        "input_files": [str(f) for f in nc_files],
        "subset_files": [str(f) for f in subset_files],
        "combined_netcdf": str(combined_nc_path),
        "extraction_time": datetime.now().isoformat(),
    }

    metadata_file = output_dir / "extraction_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    elapsed_time = time.time() - start_time
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{timestamp} - Incremental extraction complete!")
    print(f"Total time: {elapsed_time:.2f} seconds")
    print(f"Outputs saved to: {output_path}")
    print("\nFiles created:")
    print(f"  - Face subset files: {len(subset_files)} files in face_subset_files/")
    print(f"  - Combined NetCDF: {combined_nc_path}")
    print(f"  - Parquet files: {output_dir}/parquet_files/")


# Example usage and CLI integration
if __name__ == "__main__":
    # You would import your actual config here
    from config import config

    location_key = "puget_sound"
    output_path = Path("/projects/hindcastra/Tidal/datasets/projects/katie_puget_sound")
    output_path.mkdir(parents=True, exist_ok=True)
    n_closest = 10

    # Coordinates from Katie: 48°09'14.0"N 122°46'26.8"W
    lat = 48.1538889
    lon = -122.7741111

    extract_point_data_incremental(
        config,
        location_key,
        lat,
        lon,
        output_path,
        10,
    )

    # extract_point_data(
    #     config=config,
    #     location_key=location_key,
    #     target_lat=lat,
    #     target_lon=lon,
    #     output_path=Path(
    #         "/projects/hindcastra/Tidal/datasets/for_project/katie_puget_sound"
    #     ),
    #     n_closest=n_closest,
    # )
