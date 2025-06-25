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
    convert_combined_to_parquet(
        str(combined_nc_path),
        closest_faces,
        str(output_dir),
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
