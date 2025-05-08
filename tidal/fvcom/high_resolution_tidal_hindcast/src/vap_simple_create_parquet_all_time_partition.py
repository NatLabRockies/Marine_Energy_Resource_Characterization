from pathlib import Path
import os
import time
from datetime import datetime

import h5py
import numpy as np
import pandas as pd
import psutil

from . import file_manager, file_name_convention_manager


def get_partition_path(df) -> str:
    """
    Generate the partition path based on lat/lon coordinates.
    """
    lat = df["lat"].iloc[0]
    lon = df["lon"].iloc[0]
    lat_deg = int(lat)
    lon_deg = int(lon)
    lat_dec = int(abs(lat * 100) % 100)
    lon_dec = int(abs(lon * 100) % 100)

    return f"lat_deg={lat_deg:02d}/lon_deg={lon_deg:02d}/lat_dec={lat_dec:02d}/lon_dec={lon_dec:02d}"


def get_partition_file_name(
    index: int,
    df,
    config,
    location,
    index_max_digits=6,
    coord_digits_max=7,
) -> str:
    """
    Generate standardized filename for partition files.
    """
    # Round latitude and longitude to specified decimal places
    lat_rounded = round(df["lat"].iloc[0], coord_digits_max)
    lon_rounded = round(df["lon"].iloc[0], coord_digits_max)

    # Determine temporal string based on expected_delta_t_seconds
    expected_delta_t_seconds = location["expected_delta_t_seconds"]
    temporal_mapping = {3600: "1h", 1800: "30m"}
    if expected_delta_t_seconds not in temporal_mapping:
        raise ValueError(
            f"Unexpected expected_delta_t_seconds configuration {expected_delta_t_seconds}"
        )
    temporal_string = temporal_mapping[expected_delta_t_seconds]

    # Format strings for padding and precision
    index_format = f"0{index_max_digits}d"
    coord_format = f".{coord_digits_max}f"

    # Use file name convention manager to generate standard filename
    return file_name_convention_manager.generate_filename_for_data_level(
        df,
        location["output_name"],
        f"{config['dataset']['name']}.face={index:{index_format}}.lat={lat_rounded:{coord_format}}.lon={lon_rounded:{coord_format}}",
        "b4",
        temporal=temporal_string,
        ext="parquet",
    )


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


def convert_h5_to_parquet_batched(
    input_dir, output_dir, config, location, batch_size=20000
):
    """
    Convert h5 files to individual parquet files for each face using an optimized batch approach.

    This approach:
    1. Sorts input files by name (assumes time-ordered, e.g., monthly files)
    2. Uses batch reading for each variable (reading all faces in batch at once)
    3. Combines data into a single DataFrame with all time points (e.g., a full year)
    4. Writes one parquet file per face with the complete time series
    5. Includes element corner coordinates from the nv, lat_node, and lon_node datasets
    6. Organizes files in partitioned directory structure based on lat/lon

    Parameters:
    -----------
    input_dir : str
        Directory containing h5 files
    output_dir : str
        Directory to save parquet files
    config : dict
        Configuration dictionary
    location : dict
        Location configuration
    batch_size : int
        Number of faces to process in each batch
    """
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(
        f"{timestamp} - INFO - Starting h5 to parquet conversion with optimized batch approach"
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
    for i, file in enumerate(h5_files):
        print(f"  File {i+1}: {file.name}")

    # Get dataset information from the first file
    print(f"{timestamp} - INFO - Reading dataset information from first file")
    dataset_info = get_dataset_info(h5_files[0])
    total_faces = dataset_info["total_faces"]

    print(f"{timestamp} - INFO - Total faces to process: {total_faces}")
    print(f"{timestamp} - INFO - 2D datasets: {dataset_info['2d_datasets']}")
    print(
        f"{timestamp} - INFO - 3D datasets: {[(name, layers) for name, layers in dataset_info['3d_datasets']]}"
    )

    # First, read the node indices and coordinates from the first file
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{timestamp} - INFO - Reading element corner coordinates")

    # Dictionary to store node indices and coordinates for all faces
    element_corners = {}

    with h5py.File(h5_files[0], "r") as f:
        print(f"{timestamp} - INFO - Reading lat_node and lon_node arrays")
        # Read node coordinates
        lat_node = f["lat_node"][:]
        lon_node = f["lon_node"][:]
        print(f"{timestamp} - INFO - Read {len(lat_node)} node coordinates")

        print(f"{timestamp} - INFO - Reading node indices (nv array)")
        # Read node indices for each face (first time step is sufficient)
        # Note: nv is (time, 3, face) where 3 is the number of corners in each triangular face
        nv = f["nv"][0, :, :]  # Get the first time step
        print(f"{timestamp} - INFO - nv shape: {nv.shape}")

        # Convert from Fortran 1-based indexing to Python 0-based indexing
        print(f"{timestamp} - INFO - Converting from 1-based to 0-based indexing")
        nv = nv - 1

        # For each face, get the corner coordinates
        print(f"{timestamp} - INFO - Calculating corner coordinates for all faces")
        for face_id in range(total_faces):
            if face_id % 100000 == 0 and face_id > 0:
                print(
                    f"{timestamp} - INFO - Processed corner coordinates for {face_id} faces"
                )

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
        print(
            f"{timestamp} - INFO - Completed corner coordinates for all {total_faces} faces"
        )

    # Process faces in batches
    total_batches = (total_faces + batch_size - 1) // batch_size
    print(
        f"{timestamp} - INFO - Will process faces in {total_batches} batches of up to {batch_size} faces each"
    )

    for batch_idx, start_face in enumerate(range(0, total_faces, batch_size)):
        batch_start_time = time.time()
        end_face = min(start_face + batch_size, total_faces)
        faces_to_process = end_face - start_face

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(
            f"{timestamp} - INFO - Batch {batch_idx+1}/{total_batches}: Processing faces {start_face} to {end_face-1} ({faces_to_process} faces)"
        )

        # Data structure to store all time series for each face
        all_face_data = {}

        # Initialize the data structure for each face
        print(
            f"{timestamp} - INFO - Initializing data structure for {faces_to_process} faces"
        )
        for face_id in range(start_face, end_face):
            all_face_data[face_id] = {
                "time": [],
                "lat": None,
                "lon": None,
                # Element corner coordinates (constant)
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

            # Initialize dataset arrays
            for dataset_name in dataset_info["2d_datasets"]:
                if dataset_name not in ["lat_center", "lon_center"]:
                    all_face_data[face_id][dataset_name] = []

            for dataset_name, num_layers in dataset_info["3d_datasets"]:
                for layer_idx in range(num_layers):
                    col_name = f"{dataset_name}_layer_{layer_idx}"
                    all_face_data[face_id][col_name] = []

        # Process each file sequentially
        for file_idx, h5_file in enumerate(h5_files):
            file_start_time = time.time()
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(
                f"{timestamp} - INFO - File {file_idx+1}/{len(h5_files)}: Reading {h5_file}"
            )

            with h5py.File(h5_file, "r") as f:
                # Get time values once
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"{timestamp} - INFO - Reading time values")
                time_values = f["time"][:]
                time_length = len(time_values)
                print(f"{timestamp} - INFO - Found {time_length} time values")

                # Get lat/lon for all faces in this batch (only from first file)
                if file_idx == 0:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    print(f"{timestamp} - INFO - Reading lat/lon center values")

                    if "lat_center" in f:
                        print(
                            f"{timestamp} - INFO - Reading lat_center[{start_face}:{end_face}]"
                        )
                        lat_values = f["lat_center"][start_face:end_face]
                        print(
                            f"{timestamp} - INFO - Read {len(lat_values)} lat_center values"
                        )
                        for i, face_id in enumerate(range(start_face, end_face)):
                            all_face_data[face_id]["lat"] = lat_values[i]

                    if "lon_center" in f:
                        print(
                            f"{timestamp} - INFO - Reading lon_center[{start_face}:{end_face}]"
                        )
                        lon_values = f["lon_center"][start_face:end_face]
                        print(
                            f"{timestamp} - INFO - Read {len(lon_values)} lon_center values"
                        )
                        for i, face_id in enumerate(range(start_face, end_face)):
                            all_face_data[face_id]["lon"] = lon_values[i]

                # Add time values to each face
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"{timestamp} - INFO - Adding time values to all faces")
                for face_id in range(start_face, end_face):
                    all_face_data[face_id]["time"].extend(time_values)

                # Efficiently read 2D datasets in batches
                for dataset_name in dataset_info["2d_datasets"]:
                    if dataset_name in f and dataset_name not in [
                        "lat_center",
                        "lon_center",
                        "nv",
                    ]:
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        print(
                            f"{timestamp} - INFO - Reading {dataset_name}[:, {start_face}:{end_face}] (batch)"
                        )

                        # Read the entire batch of faces for all time points at once
                        batch_read_start = time.time()
                        data_batch = f[dataset_name][:, start_face:end_face]
                        batch_read_time = time.time() - batch_read_start

                        print(
                            f"{timestamp} - INFO - Read {dataset_name} batch with shape {data_batch.shape} in {batch_read_time:.2f} seconds"
                        )

                        # Distribute the data to each face
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        print(
                            f"{timestamp} - INFO - Distributing {dataset_name} data to individual faces"
                        )
                        distribute_start = time.time()

                        for i, face_id in enumerate(range(start_face, end_face)):
                            # Extract this face's data (all time points)
                            face_data = data_batch[:, i]
                            all_face_data[face_id][dataset_name].extend(face_data)

                        distribute_time = time.time() - distribute_start
                        print(
                            f"{timestamp} - INFO - Distributed {dataset_name} data in {distribute_time:.2f} seconds"
                        )

                # Efficiently read 3D datasets in batches
                for dataset_name, num_layers in dataset_info["3d_datasets"]:
                    if dataset_name in f and dataset_name not in ["nv"]:
                        for layer_idx in range(num_layers):
                            col_name = f"{dataset_name}_layer_{layer_idx}"
                            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            print(
                                f"{timestamp} - INFO - Reading {dataset_name}[:, {layer_idx}, {start_face}:{end_face}] (batch)"
                            )

                            # Read the entire batch of faces for this layer and all time points
                            batch_read_start = time.time()
                            data_batch = f[dataset_name][
                                :, layer_idx, start_face:end_face
                            ]
                            batch_read_time = time.time() - batch_read_start

                            print(
                                f"{timestamp} - INFO - Read {dataset_name} layer {layer_idx} batch with shape {data_batch.shape} in {batch_read_time:.2f} seconds"
                            )

                            # Distribute the data to each face
                            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            print(
                                f"{timestamp} - INFO - Distributing {dataset_name} layer {layer_idx} data to individual faces"
                            )
                            distribute_start = time.time()

                            for i, face_id in enumerate(range(start_face, end_face)):
                                # Extract this face's data for this layer (all time points)
                                face_data = data_batch[:, i]
                                all_face_data[face_id][col_name].extend(face_data)

                            distribute_time = time.time() - distribute_start
                            print(
                                f"{timestamp} - INFO - Distributed {dataset_name} layer {layer_idx} data in {distribute_time:.2f} seconds"
                            )

            file_time = time.time() - file_start_time
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(
                f"{timestamp} - INFO - Completed reading file {file_idx+1}/{len(h5_files)} in {file_time:.2f} seconds"
            )

        # Now write one parquet file per face with the complete time series
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(
            f"{timestamp} - INFO - Writing {faces_to_process} parquet files with full time series"
        )
        writing_start = time.time()

        for face_idx, face_id in enumerate(range(start_face, end_face)):
            if face_idx > 0 and face_idx % 1000 == 0:
                current_time = time.time()
                elapsed = current_time - writing_start
                remaining = (elapsed / face_idx) * (faces_to_process - face_idx)
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(
                    f"{timestamp} - INFO - Written {face_idx}/{faces_to_process} parquet files. Estimated time remaining: {remaining:.2f} seconds"
                )

            # Create a DataFrame for this face
            df_data = {}

            # Convert all lists to numpy arrays and handle constant values
            for key, value in all_face_data[face_id].items():
                if "nv_layer" in key:
                    continue
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
                    time_length = len(all_face_data[face_id]["time"])
                    df_data[key] = np.repeat(value, time_length)
                else:
                    df_data[key] = np.array(value)

            # for key, value in df_data.items():
            #     print(
            #         f"{key}: {value.shape}, type: {type(value)}, first five: {value[:5]}"
            #     )

            # Create DataFrame and write to parquet
            df = pd.DataFrame(df_data)
            df["time"] = pd.to_datetime(df["time"], unit="s", origin="unix")
            df = df.set_index("time")
            df = df.sort_index()

            # Create partitioned directory structure and filename
            partition_dir = Path(output_dir, get_partition_path(df))
            partition_dir.mkdir(parents=True, exist_ok=True)

            output_file = Path(
                partition_dir, get_partition_file_name(face_id, df, config, location)
            )

            df.to_parquet(output_file)

        writing_time = time.time() - writing_start
        batch_time = time.time() - batch_start_time
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(
            f"{timestamp} - INFO - Wrote {faces_to_process} parquet files in {writing_time:.2f} seconds"
        )
        print(
            f"{timestamp} - INFO - Completed batch {batch_idx+1}/{total_batches} in {batch_time:.2f} seconds"
        )

        # Calculate memory usage
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        print(f"{timestamp} - INFO - Current memory usage: {memory_mb:.2f} MB")

    elapsed_time = time.time() - start_time
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(
        f"{timestamp} - INFO - Conversion complete! Total time: {elapsed_time:.2f} seconds"
    )


def partition_vap_into_parquet_dataset(config, location_key, batch_size=20000):
    """
    Process VAP data and convert to partitioned Parquet files using an optimized batch approach.

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

    convert_h5_to_parquet_batched(
        input_path, output_path, config, location, batch_size=100000
    )
