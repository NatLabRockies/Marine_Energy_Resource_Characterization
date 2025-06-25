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
    input_files: List[Path],
    face_indices: List[int],
    output_dir: str,
    config: Dict,
    location: Dict,
) -> None:
    """
    Extract specified faces and save as individual parquet files using the existing logic.
    This is a simplified version of the batched conversion.
    """
    print(f"Extracting {len(face_indices)} faces to parquet files")

    # Get dataset info from first file
    print("Reading dataset information...")
    dataset_info = get_dataset_info(input_files[0])

    # Read element corners for the selected faces
    print("Reading element corner coordinates...")
    element_corners = {}

    with h5py.File(input_files[0], "r") as f:
        lat_node = f["lat_node"][:]
        lon_node = f["lon_node"][:]
        nv = f["nv"][0, :, :] - 1  # Convert to 0-based indexing

        for face_id in face_indices:
            node_indices = nv[:, face_id]
            element_corners[face_id] = {
                "element_corner_1_lat": lat_node[node_indices[0]],
                "element_corner_1_lon": lon_node[node_indices[0]],
                "element_corner_2_lat": lat_node[node_indices[1]],
                "element_corner_2_lon": lon_node[node_indices[1]],
                "element_corner_3_lat": lat_node[node_indices[2]],
                "element_corner_3_lon": lon_node[node_indices[2]],
            }

    # Initialize data structure for each face
    print("Initializing data structures...")
    all_face_data = {}

    for face_id in face_indices:
        all_face_data[face_id] = {
            "time": [],
            "lat": None,
            "lon": None,
            **element_corners[face_id],
        }

        # Initialize dataset arrays
        for dataset_name in dataset_info["2d_datasets"]:
            if dataset_name not in ["lat_center", "lon_center"]:
                all_face_data[face_id][dataset_name] = []

        for dataset_name, num_layers in dataset_info["3d_datasets"]:
            for layer_idx in range(num_layers):
                col_name = f"{dataset_name}_layer_{layer_idx}"
                all_face_data[face_id][col_name] = []

    # Process each file
    for file_idx, h5_file in enumerate(input_files):
        print(f"Processing file {file_idx+1}/{len(input_files)}: {h5_file.name}")

        with h5py.File(h5_file, "r") as f:
            # Get time values
            time_values = f["time"][:]

            # Add time to all faces
            for face_id in face_indices:
                all_face_data[face_id]["time"].extend(time_values)

            # Get lat/lon (only from first file)
            if file_idx == 0:
                if "lat_center" in f:
                    lat_values = f["lat_center"][face_indices]
                    for i, face_id in enumerate(face_indices):
                        all_face_data[face_id]["lat"] = lat_values[i]

                if "lon_center" in f:
                    lon_values = f["lon_center"][face_indices]
                    for i, face_id in enumerate(face_indices):
                        all_face_data[face_id]["lon"] = lon_values[i]

            # Read 2D datasets
            for dataset_name in dataset_info["2d_datasets"]:
                if dataset_name in f and dataset_name not in [
                    "lat_center",
                    "lon_center",
                    "nv",
                ]:
                    data_subset = f[dataset_name][:, face_indices]
                    for i, face_id in enumerate(face_indices):
                        all_face_data[face_id][dataset_name].extend(data_subset[:, i])

            # Read 3D datasets
            for dataset_name, num_layers in dataset_info["3d_datasets"]:
                if dataset_name in f and dataset_name not in ["nv"]:
                    for layer_idx in range(num_layers):
                        col_name = f"{dataset_name}_layer_{layer_idx}"
                        data_subset = f[dataset_name][:, layer_idx, face_indices]
                        for i, face_id in enumerate(face_indices):
                            all_face_data[face_id][col_name].extend(data_subset[:, i])

    # Write parquet files
    print("Writing parquet files...")
    os.makedirs(output_dir, exist_ok=True)

    for face_id in face_indices:
        # Create DataFrame
        df_data = {}

        for key, value in all_face_data[face_id].items():
            if key in ["lat", "lon"] + [
                k for k in all_face_data[face_id].keys() if "element_corner" in k
            ]:
                # Repeat scalar values
                time_length = len(all_face_data[face_id]["time"])
                df_data[key] = np.repeat(value, time_length)
            else:
                df_data[key] = np.array(value)

        df = pd.DataFrame(df_data)
        df["time"] = pd.to_datetime(df["time"], unit="s", origin="unix")
        df = df.set_index("time").sort_index()

        # Write parquet file
        output_file = Path(output_dir) / f"face_{face_id:06d}.parquet"
        table = pa.Table.from_pandas(df)
        pq.write_table(table, output_file)

        print(f"Wrote: {output_file}")


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
    extract_faces_to_parquet(
        nc_files, closest_faces, str(parquet_dir), config, location
    )

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


def get_file_hash(file_path: Path) -> str:
    """Generate a hash of the input files to detect changes."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        # Read first and last 1MB to create a quick hash
        chunk = f.read(1024 * 1024)
        hash_md5.update(chunk)
        try:
            f.seek(-1024 * 1024, 2)  # Seek to last 1MB
            chunk = f.read(1024 * 1024)
            hash_md5.update(chunk)
        except:
            pass  # File smaller than 2MB
    return hash_md5.hexdigest()


def create_individual_face_files(
    input_files: List[Path],
    face_indices: List[int],
    output_dir: str,
    force_recreate: bool = False,
) -> List[Path]:
    """
    Create individual NetCDF files for each face, with progress tracking.

    Parameters:
    -----------
    input_files : List[Path]
        List of input NC files
    face_indices : List[int]
        Face indices to extract
    output_dir : str
        Output directory for individual face files
    force_recreate : bool
        Force recreation of existing files

    Returns:
    --------
    List[Path]
        List of created face file paths
    """
    output_dir = Path(output_dir)
    face_files_dir = output_dir / "individual_faces"
    face_files_dir.mkdir(parents=True, exist_ok=True)

    # Create metadata file to track progress
    metadata_file = face_files_dir / "processing_metadata.json"

    # Load existing metadata if it exists
    if metadata_file.exists() and not force_recreate:
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
        print("Found existing processing metadata")
    else:
        metadata = {
            "input_files_hashes": {},
            "completed_faces": [],
            "face_file_paths": {},
            "processing_started": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
        }

    # Check if input files have changed
    current_hashes = {}
    for input_file in input_files:
        current_hashes[str(input_file)] = get_file_hash(input_file)

    files_changed = current_hashes != metadata.get("input_files_hashes", {})
    if files_changed and not force_recreate:
        print("Input files have changed since last run")
        metadata["completed_faces"] = []  # Reset progress

    metadata["input_files_hashes"] = current_hashes

    created_files = []

    for i, face_id in enumerate(face_indices):
        face_filename = f"face_{face_id:06d}.nc"
        face_file_path = face_files_dir / face_filename

        # Skip if already processed and files haven't changed
        if (
            face_id in metadata["completed_faces"]
            and face_file_path.exists()
            and not force_recreate
            and not files_changed
        ):
            print(f"Face {face_id} already processed, skipping...")
            created_files.append(face_file_path)
            continue

        print(f"Processing face {face_id} ({i+1}/{len(face_indices)})")

        try:
            # Create dataset for this single face
            datasets = []

            for j, nc_file in enumerate(input_files):
                print(f"  Reading from file {j+1}/{len(input_files)}: {nc_file.name}")

                with xr.open_dataset(nc_file, engine="h5netcdf") as ds:
                    # Select only this specific face
                    face_vars = {}

                    for var_name, var in ds.data_vars.items():
                        if "face" in var.dims:
                            # Extract only this face
                            face_vars[var_name] = var.isel(
                                face=[face_indices.index(face_id)]
                            )
                        else:
                            # Keep non-face variables as-is
                            face_vars[var_name] = var

                    # Handle coordinates
                    coords = {}
                    for coord_name, coord in ds.coords.items():
                        if "face" in coord.dims:
                            coords[coord_name] = coord.isel(
                                face=[face_indices.index(face_id)]
                            )
                        else:
                            coords[coord_name] = coord

                    # Create dataset for this face
                    face_ds = xr.Dataset(face_vars, coords=coords, attrs=ds.attrs)

                    # Drop problematic variables if they exist
                    face_ds = face_ds.drop_vars(["zeta"], errors="ignore")

                    datasets.append(face_ds)

            # Concatenate along time if multiple files
            if len(datasets) > 1:
                combined_ds = xr.concat(datasets, dim="time")
            else:
                combined_ds = datasets[0]

            # Save individual face file
            print(f"  Saving {face_file_path}")
            combined_ds.to_netcdf(face_file_path, engine="h5netcdf")
            combined_ds.close()

            # Update metadata
            metadata["completed_faces"].append(face_id)
            metadata["face_file_paths"][str(face_id)] = str(face_file_path)
            metadata["last_updated"] = datetime.now().isoformat()

            # Save metadata after each face
            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)

            created_files.append(face_file_path)
            print(f"  ✓ Completed face {face_id}")

        except Exception as e:
            print(f"  ✗ Error processing face {face_id}: {e}")
            # Remove from completed list if it was there
            if face_id in metadata["completed_faces"]:
                metadata["completed_faces"].remove(face_id)
            continue

    print(f"\nCompleted processing {len(created_files)} face files")
    return created_files


def combine_face_files_to_netcdf(
    face_files: List[Path], output_file: str, skip_if_exists: bool = True
) -> None:
    """
    Combine individual face files into a single NetCDF file.

    Parameters:
    -----------
    face_files : List[Path]
        List of individual face NetCDF files
    output_file : str
        Output combined NetCDF file path
    skip_if_exists : bool
        Skip if output file already exists
    """
    if skip_if_exists and Path(output_file).exists():
        print(f"Combined NetCDF already exists: {output_file}")
        return

    print(f"Combining {len(face_files)} face files into {output_file}")

    # Open all face files and combine along face dimension
    datasets = []
    for face_file in face_files:
        print(f"Loading {face_file.name}")
        ds = xr.open_dataset(face_file, engine="h5netcdf")
        datasets.append(ds)

    # Combine along face dimension
    print("Combining datasets...")
    combined_ds = xr.concat(datasets, dim="face")

    # Save combined file
    print(f"Saving combined NetCDF to {output_file}")
    combined_ds.to_netcdf(output_file, engine="h5netcdf")

    # Clean up
    for ds in datasets:
        ds.close()
    combined_ds.close()

    print(f"✓ Combined NetCDF saved: {output_file}")


def convert_face_files_to_parquet(
    face_files: List[Path],
    face_indices: List[int],
    output_dir: str,
    config: Dict,
    location: Dict,
    skip_existing: bool = True,
) -> None:
    """
    Convert individual face NetCDF files to parquet format.

    Parameters:
    -----------
    face_files : List[Path]
        List of individual face NetCDF files
    face_indices : List[int]
        Corresponding face indices
    output_dir : str
        Output directory for parquet files
    config : Dict
        Configuration dictionary
    location : Dict
        Location configuration
    skip_existing : bool
        Skip conversion if parquet file already exists
    """
    parquet_dir = Path(output_dir) / "parquet_files"
    parquet_dir.mkdir(parents=True, exist_ok=True)

    print(f"Converting {len(face_files)} face files to parquet format")

    for i, (face_file, face_id) in enumerate(zip(face_files, face_indices)):
        output_parquet = parquet_dir / f"face_{face_id:06d}.parquet"

        if skip_existing and output_parquet.exists():
            print(f"Parquet file already exists for face {face_id}, skipping...")
            continue

        print(f"Converting face {face_id} ({i+1}/{len(face_files)})")

        try:
            # Load the face NetCDF file
            with xr.open_dataset(face_file, engine="h5netcdf") as ds:
                # Convert to DataFrame
                df = ds.to_dataframe()

                # Reset index to make time a column
                df = df.reset_index()

                # Convert time if needed
                if "time" in df.columns:
                    if not pd.api.types.is_datetime64_any_dtype(df["time"]):
                        df["time"] = pd.to_datetime(df["time"], unit="s", origin="unix")

                # Set time as index and sort
                if "time" in df.columns:
                    df = df.set_index("time").sort_index()

                # Save as parquet
                table = pa.Table.from_pandas(df)
                pq.write_table(table, output_parquet)

                print(f"  ✓ Saved {output_parquet}")

        except Exception as e:
            print(f"  ✗ Error converting face {face_id}: {e}")
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

    This version breaks the process into steps:
    1. Create individual face files
    2. Combine face files into single NetCDF
    3. Convert to parquet files

    Each step can be resumed if interrupted.
    """
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{timestamp} - Starting incremental point data extraction")
    print(f"Target: ({target_lat:.4f}, {target_lon:.4f})")
    print(f"Number of closest points: {n_closest}")
    print(f"Output path: {output_path}")

    # Get location config (you'll need to adapt this)
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

    # Step 1: Create individual face files
    print("\n" + "=" * 50)
    print("STEP 1: Creating individual face files")
    print("=" * 50)
    face_files = create_individual_face_files(
        nc_files, closest_faces, str(output_dir), force_recreate
    )

    # Step 2: Combine face files into single NetCDF
    print("\n" + "=" * 50)
    print("STEP 2: Combining face files")
    print("=" * 50)
    combined_nc_path = output_dir / f"combined_{n_closest}_faces.nc"
    combine_face_files_to_netcdf(
        face_files, str(combined_nc_path), skip_if_exists=not force_recreate
    )

    # Step 3: Convert to parquet files
    print("\n" + "=" * 50)
    print("STEP 3: Converting to parquet files")
    print("=" * 50)
    convert_face_files_to_parquet(
        face_files,
        closest_faces,
        str(output_dir),
        config,
        location,
        skip_existing=not force_recreate,
    )

    # Save final metadata
    metadata = {
        "target_lat": target_lat,
        "target_lon": target_lon,
        "n_closest": n_closest,
        "face_indices": closest_faces,
        "distances": distances,
        "input_files": [str(f) for f in nc_files],
        "face_files": [str(f) for f in face_files],
        "combined_netcdf": str(combined_nc_path),
        "extraction_time": datetime.now().isoformat(),
    }

    metadata_file = output_dir / "final_extraction_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    elapsed_time = time.time() - start_time
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{timestamp} - Incremental extraction complete!")
    print(f"Total time: {elapsed_time:.2f} seconds")
    print(f"Outputs saved to: {output_path}")
    print("\nFiles created:")
    print(f"  - Individual face files: {len(face_files)} files")
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
