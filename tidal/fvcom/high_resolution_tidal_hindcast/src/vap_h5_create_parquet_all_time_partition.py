import os

from pathlib import Path
import multiprocessing as mp
import time
from datetime import datetime

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

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
                dataset_info["total_faces"] = f["lat_center"].shape[0]  # type: ignore
            elif "lon_center" in f:
                dataset_info["total_faces"] = f["lon_center"].shape[0]  # type: ignore
            else:
                raise ValueError(
                    "Could not determine number of faces from lat_center or lon_center"
                )

            # Get time length
            if "time" in f:
                dataset_info["time_length"] = f["time"].shape[0]  # type: ignore

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


def process_face_batch(args):
    """
    Process a batch of faces

    Parameters:
    -----------
    args : tuple
        (h5_files, face_indices, dataset_info, output_dir, batch_idx)
    """
    h5_files, face_indices, dataset_info, output_dir, batch_idx = args

    # Open h5 files once per batch to reduce file handle contention
    open_files = {}
    try:
        for h5_file in h5_files:
            open_files[h5_file] = h5py.File(h5_file, "r")

        for face_idx in face_indices:
            try:
                # Pass open file handles instead of file paths
                process_single_face(open_files, face_idx, dataset_info, output_dir)
            except Exception as e:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(
                    f"{timestamp} - ERROR - Batch {batch_idx}, Error processing face {face_idx}: {e}"
                )
    finally:
        # Close all files
        for f in open_files.values():
            f.close()


def process_single_face(open_files, face_idx, dataset_info, output_dir):
    """
    Process a single face from multiple h5 files and save it as a parquet file.

    Parameters:
    -----------
    open_files : dict
        Dictionary of open h5py.File objects
    face_idx : int
        Index of the face to process
    dataset_info : dict
        Dictionary containing dataset information
    output_dir : str
        Directory to save the parquet file
    """
    try:
        # Check if output file already exists - skip if it does
        output_file = os.path.join(output_dir, f"face_{face_idx}.parquet")
        if os.path.exists(output_file):
            return

        # Initialize data dictionary
        data_dict = {"time": [], "lat": None, "lon": None}

        # Process each h5 file
        for h5_path, f in open_files.items():
            # Get time values for this file
            time_values = f["time"][:]
            data_dict["time"].extend(time_values)

            # Get lat/lon for this face (only need to do this once)
            if data_dict["lat"] is None and "lat_center" in f:
                data_dict["lat"] = f["lat_center"][face_idx]

            if data_dict["lon"] is None and "lon_center" in f:
                data_dict["lon"] = f["lon_center"][face_idx]

            # Process 2D datasets (time, face)
            for dataset_name in dataset_info["2d_datasets"]:
                if dataset_name in f:
                    if dataset_name not in data_dict:
                        data_dict[dataset_name] = []
                    data_dict[dataset_name].extend(f[dataset_name][:, face_idx])

            # Process 3D datasets (time, layer, face)
            for dataset_name, num_layers in dataset_info["3d_datasets"]:
                if dataset_name in f:
                    for layer_idx in range(num_layers):
                        col_name = f"{dataset_name}_layer_{layer_idx}"
                        if col_name not in data_dict:
                            data_dict[col_name] = []
                        data_dict[col_name].extend(
                            f[dataset_name][:, layer_idx, face_idx]
                        )

        # Create DataFrame
        df_data = {}
        time_length = len(data_dict["time"])

        for key, value in data_dict.items():
            if key in ["lat", "lon"] and value is not None:
                # Repeat scalar values to match time length
                df_data[key] = np.repeat(value, time_length)
            elif value:
                df_data[key] = value

        df = pd.DataFrame(df_data)

        # Save as parquet
        df.to_parquet(output_file)

    except Exception as e:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"{timestamp} - ERROR - Error processing face {face_idx}: {e}")
        raise


def convert_h5_to_parquet(
    input_dir, output_dir, batch_size=100, num_processes=16, max_io_processes=8
):
    """
    Convert h5 files to individual parquet files for each face.

    Parameters:
    -----------
    input_dir : str
        Directory containing h5 files
    output_dir : str
        Directory to save parquet files
    batch_size : int
        Number of faces to process in each batch
    num_processes : int
        Number of processes to use (default: 16)
    max_io_processes : int
        Maximum number of processes that can simultaneously read files
        to avoid I/O contention (default: 8)
    """
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{timestamp} - INFO - Starting h5 to parquet conversion")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get all h5 files
    h5_files = sorted(list(Path(input_dir).glob("*.h5")))
    if not h5_files:
        print(f"{timestamp} - ERROR - No h5 files found in {input_dir}")
        return

    print(f"{timestamp} - INFO - Found {len(h5_files)} h5 files")

    # Get dataset information from the first file
    dataset_info = get_dataset_info(h5_files[0])
    total_faces = dataset_info["total_faces"]

    print(f"{timestamp} - INFO - Total faces to process: {total_faces}")
    print(f"{timestamp} - INFO - 2D datasets: {dataset_info['2d_datasets']}")
    print(
        f"{timestamp} - INFO - 3D datasets: {[(name, layers) for name, layers in dataset_info['3d_datasets']]}"
    )

    # Limit the number of processes to avoid I/O contention
    if num_processes > 108:  # Your HPC has 108 cores
        print(
            f"{timestamp} - WARNING - Limiting from {num_processes} to 108 processes (max cores)"
        )
        num_processes = 108

    if num_processes > max_io_processes:
        print(
            f"{timestamp} - INFO - Limiting I/O processes to {max_io_processes} to avoid HDF5 read contention"
        )
        io_processes = max_io_processes
    else:
        io_processes = num_processes

    print(
        f"{timestamp} - INFO - Using {io_processes} I/O processes out of {num_processes} total processes"
    )

    # Create batches of face indices
    face_batches = []
    batch_idx = 0
    for start_idx in range(0, total_faces, batch_size):
        end_idx = min(start_idx + batch_size, total_faces)
        face_batches.append((list(range(start_idx, end_idx)), batch_idx))
        batch_idx += 1

    # Process batches in parallel
    args_list = [
        (h5_files, batch, dataset_info, output_dir, idx) for batch, idx in face_batches
    ]

    with mp.Pool(processes=io_processes) as pool:
        list(
            tqdm(
                pool.imap(process_face_batch, args_list),
                total=len(args_list),
                desc="Processing face batches",
                ncols=100,
            )
        )

    elapsed_time = time.time() - start_time
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(
        f"{timestamp} - INFO - Conversion complete! Total time: {elapsed_time:.2f} seconds"
    )


def partition_vap_into_parquet_dataset(config, location_key, max_workers=96):
    """
    Process VAP data and convert to partitioned Parquet files.
    Utilizes a hybrid approach for optimal performance on HPC systems.

    Parameters
    ----------
    config : dict
        Configuration dictionary
    location_key : str
        Key for location in the configuration
    max_workers : int, optional
        Maximum number of parallel processes to use (default: 96)
    """
    location = config["location_specification"][location_key]
    input_path = file_manager.get_vap_output_dir(config, location)
    output_path = file_manager.get_vap_partition_output_dir(
        config, location, use_temp_base_path=True
    )

    convert_h5_to_parquet(
        input_path,
        output_path,
        batch_size=10000,
        num_processes=96,
        max_io_processes=8,
    )

    final_output_path = file_manager.get_vap_partition_output_dir(
        config, location, use_temp_base_path=False
    )

    if output_path != final_output_path:
        print(f"Copying output files from {output_path} to {final_output_path}...")

        copy_manager.copy_directory(output_path, final_output_path)

        print(f"Copy complete! Output files are in {final_output_path}")
