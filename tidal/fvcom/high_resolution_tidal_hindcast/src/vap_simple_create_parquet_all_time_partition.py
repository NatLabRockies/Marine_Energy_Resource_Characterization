import os
from pathlib import Path
import time
from datetime import datetime

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

# Import modules from the main project
from . import copy_manager, file_manager, file_name_convention_manager


def get_dataset_info(h5_file_path):
    """
    Get information about datasets in the h5 file

    Parameters:
    -----------
    h5_file_path : str
        Path to the h5 file

    Returns:
    --------
    dict
        Dictionary containing dataset information
    """
    dataset_info = {
        "total_faces": 0,
        "2d_datasets": [],  # (time, face)
        "3d_datasets": [],  # (time, layer, face)
        "other_datasets": [],
        "time_length": 0,
    }

    try:
        with h5py.File(h5_file_path, "r") as f:
            # Get the number of faces
            if "lat_center" in f:
                dataset_info["total_faces"] = f["lat_center"].shape[0]
            elif "lon_center" in f:
                dataset_info["total_faces"] = f["lon_center"].shape[0]
            else:
                raise ValueError(
                    "Could not determine number of faces from lat_center or lon_center"
                )

            # Get time length
            if "time" in f:
                dataset_info["time_length"] = f["time"].shape[0]

            # Categorize datasets
            for dataset_name, dataset in f.items():
                if isinstance(dataset, h5py.Dataset):
                    shape = dataset.shape

                    # Skip lat/lon nodes as they're not per-face
                    if dataset_name in [
                        "lat_node",
                        "lon_node",
                        "node",
                        "face_node",
                        "face_node_index",
                    ]:
                        continue

                    # Skip single-dimension datasets that aren't time
                    if len(shape) == 1 and dataset_name != "time":
                        continue

                    # 2D datasets with (time, face)
                    if len(shape) == 2 and shape[1] == dataset_info["total_faces"]:
                        dataset_info["2d_datasets"].append(dataset_name)

                    # 3D datasets with (time, layer, face)
                    elif len(shape) == 3 and shape[2] == dataset_info["total_faces"]:
                        dataset_info["3d_datasets"].append(
                            (dataset_name, shape[1])
                        )  # Include number of layers

                    # Other datasets
                    else:
                        dataset_info["other_datasets"].append(dataset_name)

    except Exception as e:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(
            f"{timestamp} - ERROR - Error getting dataset info from {h5_file_path}: {e}"
        )
        raise

    return dataset_info


def convert_h5_to_parquet_simple(input_dir, output_dir, batch_size=20000):
    """
    Convert h5 files to individual parquet files for each face using a simple sequential approach.

    This approach:
    1. Sorts input files by name (assumes time-ordered, e.g., monthly files)
    2. Reads all files for a batch of faces
    3. Combines data into a single DataFrame with all time points (e.g., a full year)
    4. Writes one parquet file per face with the complete time series
    5. Includes element corner coordinates from the nv, lat_node, and lon_node datasets

    Parameters:
    -----------
    input_dir : str
        Directory containing h5 files
    output_dir : str
        Directory to save parquet files
    batch_size : int
        Number of faces to process in each batch
    """
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(
        f"{timestamp} - INFO - Starting h5 to parquet conversion with simple batch approach"
    )

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get all h5 files and sort them by name (assumes chronological ordering)
    h5_files = sorted(list(Path(input_dir).glob("*.nc")))
    if not h5_files:
        h5_files = sorted(list(Path(input_dir).glob("*.h5")))
        if not h5_files:
            print(f"{timestamp} - ERROR - No h5 or nc files found in {input_dir}")
            return

    print(f"{timestamp} - INFO - Found {len(h5_files)} files (sorted by name)")

    # Get dataset information from the first file
    dataset_info = get_dataset_info(h5_files[0])
    total_faces = dataset_info["total_faces"]

    print(f"{timestamp} - INFO - Total faces to process: {total_faces}")
    print(f"{timestamp} - INFO - 2D datasets: {dataset_info['2d_datasets']}")
    print(
        f"{timestamp} - INFO - 3D datasets: {[(name, layers) for name, layers in dataset_info['3d_datasets']]}"
    )

    # First, read the node indices and coordinates from the first file
    # These don't change over time, so we only need to read them once
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{timestamp} - INFO - Reading element corner coordinates")

    # Dictionary to store node indices and coordinates for all faces
    element_corners = {}

    with h5py.File(h5_files[0], "r") as f:
        # Read node coordinates
        lat_node = f["lat_node"][:]
        lon_node = f["lon_node"][:]

        # Read node indices for each face (first time step is sufficient)
        # Note: nv is (time, 3, face) where 3 is the number of corners in each triangular face
        nv = f["nv"][0, :, :]  # Get the first time step

        # Convert from Fortran 1-based indexing to Python 0-based indexing
        nv = nv - 1

        # For each face, get the corner coordinates
        for face_id in range(total_faces):
            # Get the 3 node indices for this face
            node_indices = nv[:, face_id]

            # Get the coordinates for each corner
            element_corners[face_id] = {
                "element_corner_1_lat": lat_node[node_indices[0]],
                "element_corner_1_lon": lon_node[node_indices[0]],
                "element_corner_2_lat": lat_node[node_indices[1]],
                "element_corner_2_lon": lon_node[node_indices[1]],
                "element_corner_3_lat": lat_node[node_indices[2]],
                "element_corner_3_lon": lon_node[node_indices[2]],
            }

    # Process faces in batches
    for start_face in range(0, total_faces, batch_size):
        batch_start_time = time.time()
        end_face = min(start_face + batch_size, total_faces)
        faces_to_process = end_face - start_face

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(
            f"{timestamp} - INFO - Processing faces {start_face} to {end_face-1} ({faces_to_process} faces)"
        )

        # First, collect data for all faces in this batch across all files
        # We'll organize data by face_id first
        face_data = {}

        # Initialize face_data structure
        for face_id in range(start_face, end_face):
            face_data[face_id] = {
                "time": [],
                "lat": None,
                "lon": None,
                # Add element corner coordinates (these are constant for each face)
                "element_corner_1_lat": element_corners[face_id][
                    "element_corner_1_lat"
                ],
                "element_corner_1_lon": element_corners[face_id][
                    "element_corner_1_lon"
                ],
                "element_corner_2_lat": element_corners[face_id][
                    "element_corner_2_lat"
                ],
                "element_corner_2_lon": element_corners[face_id][
                    "element_corner_2_lon"
                ],
                "element_corner_3_lat": element_corners[face_id][
                    "element_corner_3_lat"
                ],
                "element_corner_3_lon": element_corners[face_id][
                    "element_corner_3_lon"
                ],
            }

            # Initialize all dataset columns
            for dataset_name in dataset_info["2d_datasets"]:
                if dataset_name not in ["lat_center", "lon_center"]:  # Skip lat/lon
                    face_data[face_id][dataset_name] = []

            # Initialize 3D dataset columns
            for dataset_name, num_layers in dataset_info["3d_datasets"]:
                for layer_idx in range(num_layers):
                    col_name = f"{dataset_name}_layer_{layer_idx}"
                    face_data[face_id][col_name] = []

        # Process each file sequentially
        for file_idx, h5_file in enumerate(h5_files):
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(
                f"{timestamp} - INFO - Reading file {file_idx+1}/{len(h5_files)}: {h5_file}"
            )

            with h5py.File(h5_file, "r") as f:
                # Get time values
                time_values = f["time"][:]

                # Get lat/lon for all faces in this batch (only from first file)
                if file_idx == 0:
                    if "lat_center" in f:
                        lat_values = f["lat_center"][start_face:end_face]
                        for i, face_id in enumerate(range(start_face, end_face)):
                            face_data[face_id]["lat"] = lat_values[i]

                    if "lon_center" in f:
                        lon_values = f["lon_center"][start_face:end_face]
                        for i, face_id in enumerate(range(start_face, end_face)):
                            face_data[face_id]["lon"] = lon_values[i]

                # For each face, extract all data from this file
                for face_id in range(start_face, end_face):
                    # Add time values for this file
                    face_data[face_id]["time"].extend(time_values)

                    # Process 2D datasets (time, face)
                    for dataset_name in dataset_info["2d_datasets"]:
                        if dataset_name in f and dataset_name not in [
                            "lat_center",
                            "lon_center",
                        ]:
                            # Extract data for this face across all time points in this file
                            data = f[dataset_name][:, face_id]
                            face_data[face_id][dataset_name].extend(data)

                    # Process 3D datasets (time, layer, face)
                    for dataset_name, num_layers in dataset_info["3d_datasets"]:
                        if dataset_name in f:
                            for layer_idx in range(num_layers):
                                col_name = f"{dataset_name}_layer_{layer_idx}"

                                # Extract data for this face and layer across all time points
                                data = f[dataset_name][:, layer_idx, face_id]
                                face_data[face_id][col_name].extend(data)

        # Now write one parquet file per face with the complete time series
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(
            f"{timestamp} - INFO - Writing {faces_to_process} parquet files with full time series"
        )

        for face_id in range(start_face, end_face):
            # Create a DataFrame for this face
            df_data = {}

            # Convert all lists to numpy arrays and handle constant values
            for key, value in face_data[face_id].items():
                if key in [
                    "lat",
                    "lon",
                    "element_corner_1_lat",
                    "element_corner_1_lon",
                    "element_corner_2_lat",
                    "element_corner_2_lon",
                    "element_corner_3_lat",
                    "element_corner_3_lon",
                ]:
                    # Repeat scalar values to match time length
                    time_length = len(face_data[face_id]["time"])
                    df_data[key] = np.repeat(value, time_length)
                else:
                    df_data[key] = np.array(value)

            # Create DataFrame and write to parquet
            df = pd.DataFrame(df_data)

            output_file = os.path.join(output_dir, f"face_{face_id}.parquet")
            df.to_parquet(output_file)

        batch_time = time.time() - batch_start_time
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"{timestamp} - INFO - Completed batch in {batch_time:.2f} seconds")

    elapsed_time = time.time() - start_time
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(
        f"{timestamp} - INFO - Conversion complete! Total time: {elapsed_time:.2f} seconds"
    )


def partition_vap_into_parquet_dataset(config, location_key, batch_size=20000):
    """
    Process VAP data and convert to partitioned Parquet files using a simple sequential approach.

    Parameters
    ----------
    config : dict
        Configuration dictionary
    location_key : str
        Key for location in the configuration
    batch_size : int, optional
        Number of faces to process in each batch (default: 20000)
    """
    location = config["location_specification"][location_key]
    input_path = file_manager.get_vap_output_dir(config, location)
    output_path = file_manager.get_vap_partition_output_dir(
        config, location, use_temp_base_path=False
    )

    # Use simple batch approach
    convert_h5_to_parquet_simple(input_path, output_path, batch_size=batch_size)

    # final_output_path = file_manager.get_vap_partition_output_dir(
    #     config, location, use_temp_base_path=False
    # )
    #
    # if output_path != final_output_path:
    #     print(f"Copying output files from {output_path} to {final_output_path}...")
    #     copy_manager.copy_directory(output_path, final_output_path)
    #     print(f"Copy complete! Output files are in {final_output_path}")
