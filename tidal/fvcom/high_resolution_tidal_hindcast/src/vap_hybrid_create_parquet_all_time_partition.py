import concurrent.futures
import multiprocessing as mp
import json
import os
import math
import time

from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import xarray as xr
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from . import copy_manager, file_manager, file_name_convention_manager, nc_manager


def standalone_save_parquet_batch(args, safe_config, safe_location):
    """
    Standalone function to process and save a batch of face data to Parquet files.

    Parameters
    ----------
    args : tuple
        Tuple containing batch processing arguments
    safe_config : dict
        Simplified config dictionary with only serializable objects
    safe_location : dict
        Simplified location dictionary with only serializable objects

    Returns
    -------
    dict
        Statistics about the processed batch
    """
    start_time = time.time()
    (
        face_data_batch,
        face_indices,
        vars_to_include,
        attributes,
        output_dir,
        write_batch_size,
    ) = args

    worker_id = os.getpid()
    print(
        f"Worker {worker_id}: Starting at {time.strftime('%H:%M:%S')} - Processing batch of {len(face_indices)} faces"
    )

    # Convert output_dir back to Path if it's a string
    if isinstance(output_dir, str):
        output_dir = Path(output_dir)

    # Extract face data into DataFrames
    face_dfs = {}
    time_values = face_data_batch["time_values"]
    time_dim_len = len(time_values)

    # Variables to skip in extraction
    vars_to_skip = ["nv", "h_center"]

    # Process each face
    for i, face_idx in enumerate(face_indices):
        local_idx = i  # Use index in the provided subset data
        data_dict = {}

        # Add center coordinates
        data_dict["lat_center"] = [
            face_data_batch["lat_center"][local_idx]
        ] * time_dim_len
        data_dict["lon_center"] = [
            face_data_batch["lon_center"][local_idx]
        ] * time_dim_len

        # Process node vertex data
        nv_data = face_data_batch["nv"][local_idx]
        node_indices = [int(idx - 1) if idx > 0 else int(idx) for idx in nv_data]

        # Add corner node data if available
        if "lat_node" in face_data_batch and "lon_node" in face_data_batch:
            for j, node_idx in enumerate(node_indices):
                corner_num = j + 1
                lat_node_val = float(face_data_batch["lat_node"][node_idx])
                lon_node_val = float(face_data_batch["lon_node"][node_idx])

                data_dict[f"element_corner_{corner_num}_lat"] = np.repeat(
                    lat_node_val, time_dim_len
                )
                data_dict[f"element_corner_{corner_num}_lon"] = np.repeat(
                    lon_node_val, time_dim_len
                )

        # Add variable data for this face
        for var_name in vars_to_include:
            if var_name in vars_to_skip or var_name not in face_data_batch:
                continue

            if f"{var_name}_has_layers" in face_data_batch:
                # Handle layered variables
                for layer_idx in range(face_data_batch[f"{var_name}_num_layers"]):
                    col_name = f"{var_name}_layer_{layer_idx}"
                    var_data = face_data_batch[f"{var_name}_layer_{layer_idx}"][
                        :, local_idx
                    ]

                    # Verify data length
                    if len(var_data) != time_dim_len:
                        print(
                            f"Warning: {col_name} for face {face_idx} has time dimension {len(var_data)} != {time_dim_len}"
                        )
                        continue

                    data_dict[col_name] = var_data

            elif var_name in face_data_batch:
                # Handle 2D variables
                var_data = face_data_batch[var_name][:, local_idx]

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
        face_dfs[face_idx] = df

    # Prepare write tasks
    all_write_tasks = []
    for face_idx, df in face_dfs.items():
        lat = float(face_data_batch["lat_center_all"][face_idx])
        lon = float(face_data_batch["lon_center_all"][face_idx])

        # Generate partition path and create directory
        lat_deg = int(lat)
        lon_deg = int(lon)
        lat_dec = int(abs(lat * 100) % 100)
        lon_dec = int(abs(lon * 100) % 100)

        partition_path = f"lat_deg={lat_deg:02d}/lon_deg={lon_deg:02d}/lat_dec={lat_dec:02d}/lon_dec={lon_dec:02d}"
        full_dir = Path(output_dir, partition_path)
        full_dir.mkdir(exist_ok=True, parents=True)

        # Generate filename
        if safe_config is not None and safe_location is not None:
            # Use simplified logic to generate filename
            filename = f"face_{face_idx}.parquet"

            # If config contains necessary info for custom filename
            if (
                "dataset" in safe_config
                and "name" in safe_config["dataset"]
                and "output_name" in safe_location
            ):
                # Format a simplified filename
                dataset_name = safe_config["dataset"]["name"]
                lat_rounded = round(lat, 7)
                lon_rounded = round(lon, 7)
                filename = f"{dataset_name}.face={face_idx:06d}.lat={lat_rounded:.7f}.lon={lon_rounded:.7f}.parquet"
        else:
            filename = f"face_{face_idx}.parquet"

        all_write_tasks.append((df, attributes, partition_path, filename, face_idx))

    # Process write tasks in smaller parallel batches
    saved_paths = []
    partitions_created = set()

    # Report progress at regular intervals
    total_batches = (len(all_write_tasks) + write_batch_size - 1) // write_batch_size
    report_every = max(1, total_batches // 5)  # Report ~5 times during processing

    for i in range(0, len(all_write_tasks), write_batch_size):
        batch_num = i // write_batch_size + 1
        batch_tasks = all_write_tasks[i : i + write_batch_size]

        # Process each task in this batch
        batch_paths = []
        for df, attributes, partition_path, filename, face_idx in batch_tasks:
            # Prepare file path
            full_dir = Path(output_dir, partition_path)
            full_path = Path(full_dir, filename)

            # Convert to PyArrow table with metadata
            table = pa.Table.from_pandas(df)

            # Prepare metadata
            metadata = {}
            if "variable_attributes" in attributes:
                for var_name, var_attrs in attributes["variable_attributes"].items():
                    for attr_name, attr_value in var_attrs.items():
                        metadata[f"{var_name}:{attr_name}"] = attr_value

            if "global_attributes" in attributes:
                for attr_name, attr_value in attributes["global_attributes"].items():
                    metadata[f"global:{attr_name}"] = attr_value

            # Add metadata markers
            metadata["_WPTO_HINDCAST_FORMAT_VERSION"] = "1.0"
            metadata["_WPTO_HINDCAST_METADATA_TYPE"] = "netcdf_compatible"

            # Convert all metadata values to bytes
            metadata_bytes = {}
            for k, v in metadata.items():
                try:
                    if (
                        isinstance(v, (list, dict, tuple))
                        or hasattr(v, "__dict__")
                        or isinstance(v, np.ndarray)
                    ):
                        metadata_bytes[k] = json.dumps(v, cls=json.JSONEncoder).encode(
                            "utf-8"
                        )
                    else:
                        metadata_bytes[k] = str(v).encode("utf-8")
                except TypeError:
                    metadata_bytes[k] = str(v).encode("utf-8")

            # Apply metadata
            table = table.replace_schema_metadata(
                {**table.schema.metadata, **metadata_bytes}
            )

            # Write the file
            pq.write_table(table, full_path)
            batch_paths.append(str(full_path))

            # Track partitions created
            partitions_created.add(str(Path(partition_path)))

        saved_paths.extend(batch_paths)

        # Report progress only at specific intervals
        if batch_num % report_every == 0 or batch_num == total_batches:
            progress = int((batch_num / total_batches) * 100)
            print(
                f"Worker {worker_id}: {progress}% complete ({batch_num}/{total_batches} batches)"
            )

    elapsed_time = time.time() - start_time
    print(
        f"Worker {worker_id}: Completed at {time.strftime('%H:%M:%S')} - Processed {len(face_indices)} faces in {elapsed_time:.2f} seconds"
    )

    return {
        "worker_id": worker_id,
        "faces_processed": len(face_indices),
        "files_created": len(saved_paths),
        "partitions_created": list(partitions_created),
        "elapsed_time": elapsed_time,
    }


class ConvertTidalNcToParquet:
    """
    Converts an xarray Dataset with FVCOM structure to partitioned Parquet files.
    Each face in the dataset will be converted to a time-indexed Parquet file
    stored in a partition based on its lat/lon coordinates.

    Optimized for HPC environments with many cores and large memory,
    with a focus on efficient NetCDF file access.
    """

    def __init__(self, output_dir: str, config=None, location=None, max_workers=96):
        """
        Initialize the converter with optimized parallelism settings.

        Parameters
        ----------
        output_dir : str
            Directory where the output files will be stored
        config : dict, optional
            Configuration dictionary
        location : dict, optional
            Location specification from the config
        max_workers : int, optional
            Maximum number of parallel processes to use (default: 96)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.config = config
        self.location = location
        self.max_workers = max_workers
        self.manager = mp.Manager()
        self.output_path_map = self.manager.dict()  # Shared dictionary for output paths
        self.lock = self.manager.Lock()  # Lock for thread-safe operations

    @staticmethod
    def _get_partition_path(lat: float, lon: float) -> str:
        """
        Generate the partition path based on lat/lon coordinates.
        """
        lat_deg = int(lat)
        lon_deg = int(lon)
        lat_dec = int(abs(lat * 100) % 100)
        lon_dec = int(abs(lon * 100) % 100)

        return f"lat_deg={lat_deg:02d}/lon_deg={lon_deg:02d}/lat_dec={lat_dec:02d}/lon_dec={lon_dec:02d}"

    def _get_partition_file_name(
        self,
        index: int,
        lat: float,
        lon: float,
        df,
        index_max_digits=6,
        coord_digits_max=7,
    ) -> str:
        """
        Generate standardized filename for partition files.
        """
        # Round latitude and longitude to specified decimal places
        lat_rounded = round(lat, coord_digits_max)
        lon_rounded = round(lon, coord_digits_max)

        # Determine temporal string based on expected_delta_t_seconds
        expected_delta_t_seconds = self.location["expected_delta_t_seconds"]
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
            self.location["output_name"],
            f"{self.config['dataset']['name']}.face={index:{index_format}}.lat={lat_rounded:{coord_format}}.lon={lon_rounded:{coord_format}}",
            "b4",
            temporal=temporal_string,
            ext="parquet",
        )

    @staticmethod
    def _extract_attributes(dataset: xr.Dataset) -> Dict:
        """
        Extract global and variable attributes from the dataset.
        """
        # Get global attributes
        global_attrs = dict(dataset.attrs)

        # Get variable attributes
        var_attrs = {}
        for var_name, var in dataset.variables.items():
            var_attrs[var_name] = dict(var.attrs)

        return {"global_attrs": global_attrs, "variable_attrs": var_attrs}

    @staticmethod
    def _prepare_netcdf_compatible_metadata(attributes: Dict) -> Dict:
        """
        Process and prepare metadata to be compatible with NetCDF/xarray structure.
        """

        # Custom JSON encoder for NumPy types
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
                    return int(obj)
                elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
                    return float(obj)
                elif isinstance(obj, (np.ndarray,)):
                    return obj.tolist()
                elif isinstance(obj, (np.bool_)):
                    return bool(obj)
                elif np.isnan(obj):
                    return None
                elif hasattr(obj, "isoformat"):  # datetime objects
                    return obj.isoformat()
                return super(NumpyEncoder, self).default(obj)

        metadata = {}

        # Process attributes based on structure
        if "variable_attributes" in attributes:
            for var_name, var_attrs in attributes["variable_attributes"].items():
                for attr_name, attr_value in var_attrs.items():
                    metadata[f"{var_name}:{attr_name}"] = attr_value

        if "global_attributes" in attributes:
            for attr_name, attr_value in attributes["global_attributes"].items():
                metadata[f"global:{attr_name}"] = attr_value

        # Handle flat dictionary case
        if not isinstance(attributes, dict) or (
            "variable_attributes" not in attributes
            and "global_attributes" not in attributes
        ):
            for attr_name, attr_value in attributes.items():
                metadata[f"global:{attr_name}"] = attr_value

        # Add metadata markers
        metadata["_WPTO_HINDCAST_FORMAT_VERSION"] = "1.0"
        metadata["_WPTO_HINDCAST_METADATA_TYPE"] = "netcdf_compatible"

        # Convert all metadata values to bytes
        metadata_bytes = {}
        for k, v in metadata.items():
            try:
                if (
                    isinstance(v, (list, dict, tuple))
                    or hasattr(v, "__dict__")
                    or isinstance(v, np.ndarray)
                ):
                    metadata_bytes[k] = json.dumps(v, cls=NumpyEncoder).encode("utf-8")
                else:
                    metadata_bytes[k] = str(v).encode("utf-8")
            except TypeError as e:
                metadata_bytes[k] = str(v).encode("utf-8")
                print(f"Warning: Could not JSON encode {k}: {e}")

        return metadata_bytes

    def _load_face_data(
        self, dataset: xr.Dataset, face_indices: List[int], vars_to_include: List[str]
    ) -> Dict:
        """
        Extract and prepare data for a batch of faces from the NetCDF dataset.
        This function is called by the main process to prepare data for workers.

        Parameters
        ----------
        dataset : xr.Dataset
            The NetCDF dataset
        face_indices : List[int]
            Indices of faces to extract
        vars_to_include : List[str]
            Variables to include in the extraction

        Returns
        -------
        Dict
            Dictionary containing pre-loaded data for the specified faces
        """
        batch_data = {
            "time_values": dataset.time.values,
            "lat_center": dataset.lat_center.values[face_indices],
            "lon_center": dataset.lon_center.values[face_indices],
            "lat_center_all": dataset.lat_center.values,  # For partitioning in workers
            "lon_center_all": dataset.lon_center.values,  # For partitioning in workers
            "nv": dataset["nv"].isel(time=0).isel(face=face_indices).values.T,
        }

        # Pre-fetch lat_node and lon_node if they exist
        if "lat_node" in dataset.variables and "lon_node" in dataset.variables:
            batch_data["lat_node"] = dataset["lat_node"].values
            batch_data["lon_node"] = dataset["lon_node"].values

        # Variables to skip in extraction
        vars_to_skip = ["nv", "h_center"]

        # Pre-fetch all variable data
        for var_name in vars_to_include:
            if var_name in vars_to_skip:
                continue

            var = dataset[var_name]

            # Extract data based on variable dimensions
            if "sigma_layer" in var.dims and "face" in var.dims and "time" in var.dims:
                # 3D variables (time, sigma_layer, face)
                selected_data = var.isel(face=face_indices)

                batch_data[f"{var_name}_has_layers"] = [
                    "sigma_layer"
                ]  # Mark as layered
                batch_data[f"{var_name}_num_layers"] = len(dataset.sigma_layer)

                for layer_idx in range(len(dataset.sigma_layer)):
                    layer_data = selected_data.isel(sigma_layer=layer_idx)

                    # Ensure time dimension is first
                    if layer_data.dims[0] == "time" and layer_data.dims[1] == "face":
                        data_array = layer_data.values
                    else:
                        data_array = layer_data.transpose("time", "face").values

                    batch_data[f"{var_name}_layer_{layer_idx}"] = data_array

            elif "face" in var.dims and "time" in var.dims:
                # 2D variables (time, face)
                faces_data = var.isel(face=face_indices)

                # Ensure time dimension is first
                if "time" in faces_data.dims:
                    time_dim_idx = faces_data.dims.index("time")
                    if time_dim_idx != 0:
                        dim_order = list(faces_data.dims)
                        dim_order.remove("time")
                        dim_order.insert(0, "time")
                        faces_data = faces_data.transpose(*dim_order)

                batch_data[var_name] = faces_data.values

        return batch_data

    def convert_dataset(
        self,
        dataset_path: Path,
        vars_to_include: Optional[List[str]] = None,
        max_faces: Optional[int] = None,
        write_batch_size: int = 16,
        main_batch_size: int = 100000,
    ) -> Dict:
        """
        Convert an xarray Dataset to partitioned Parquet files.
        Uses a hybrid approach: main process reads the NetCDF file,
        worker processes handle data transformation and file writing.

        Parameters
        ----------
        dataset_path : Path
            Path to the NetCDF dataset
        vars_to_include : List[str], optional
            List of variable names to include. If None, includes all variables.
        max_faces : int, optional
            Maximum number of faces to process
        write_batch_size : int, optional
            Number of files to write in parallel within each process
        main_batch_size : int, optional
            Number of faces to load at once in the main process

        Returns
        -------
        Dict
            Statistics about the conversion process
        """
        print(f"Starting conversion of {dataset_path}")
        start_time = time.time()

        # Open and analyze the dataset in the main process
        print("Loading and analyzing dataset")
        dataset = nc_manager.nc_open(dataset_path, self.config)

        # Extract attributes and determine variables to include
        attributes = self._extract_attributes(dataset)
        if vars_to_include is None:
            vars_to_include = list(dataset.variables.keys())

        # Determine number of faces to process
        num_faces = len(dataset.lat_center.values)
        if max_faces is not None:
            num_faces = min(num_faces, max_faces)

        # Determine optimal batch sizes and worker count
        num_workers = min(
            self.max_workers, mp.cpu_count(), math.ceil(num_faces / 100)
        )  # At least 100 faces per worker

        worker_batch_size = min(500, math.ceil(num_faces / num_workers))
        num_batches = math.ceil(num_faces / main_batch_size)

        # Get output_dir as a string to avoid serialization issues
        output_dir_str = str(self.output_dir)

        print("CONVERSION PLAN:")
        print(f"- Total faces: {num_faces}")
        print(f"- Main process batch size: {main_batch_size}")
        print(f"- Using up to {num_workers} worker processes")
        print(f"- Worker batch size: ~{worker_batch_size} faces")
        print(f"- Write batch size: {write_batch_size}")
        print(f"- Total main process batches: {num_batches}")

        # Initialize combined statistics
        combined_stats = {
            "total_faces": num_faces,
            "workers_used": num_workers,
            "partitions_created": set(),
            "files_created": 0,
        }

        # Process the dataset in batches to manage memory usage
        for batch_idx in range(num_batches):
            batch_start = batch_idx * main_batch_size
            batch_end = min(batch_start + main_batch_size, num_faces)
            batch_faces = list(range(batch_start, batch_end))
            batch_size = len(batch_faces)

            print(
                f"\nProcessing main batch {batch_idx + 1}/{num_batches} ({batch_size} faces)"
            )

            print("Loading face data...")
            face_load_start = time.time()
            # Load this batch of data in the main process
            face_data_batch = self._load_face_data(
                dataset, batch_faces, vars_to_include
            )

            print(f"Face data loaded in {time.time() - face_load_start:.2f} seconds")

            # Clean the config and location dictionaries to remove any unpicklable objects
            # This creates simplified versions that can be safely passed to worker processes
            safe_config = {}  # type: ignore
            if self.config is not None:
                for key, value in self.config.items():
                    if (
                        isinstance(value, (str, int, float, bool, list, dict))
                        or value is None
                    ):
                        if isinstance(value, dict):
                            # Recursively clean nested dictionaries
                            safe_config[key] = {}
                            for k, v in value.items():
                                if (
                                    isinstance(v, (str, int, float, bool, list, dict))
                                    or v is None
                                ):
                                    safe_config[key][k] = v
                        else:
                            safe_config[key] = value

            safe_location = {}
            if self.location is not None:
                for key, value in self.location.items():
                    if (
                        isinstance(value, (str, int, float, bool, list, dict))
                        or value is None
                    ):
                        safe_location[key] = value

            # Divide the loaded batch into worker sub-batches with optimized data transfer
            print("Creating optimized data subsets for each worker...")
            worker_batches = []
            for i in range(0, batch_size, worker_batch_size):
                end_idx = min(i + worker_batch_size, batch_size)

                # Get the actual face indices this worker will process
                sub_batch_indices = batch_faces[i:end_idx]

                # Calculate local indices within the current main batch
                local_start = i
                local_end = end_idx
                local_size = local_end - local_start

                # Create a subset of the data just for this worker's faces
                worker_data_batch = {
                    "time_values": face_data_batch[
                        "time_values"
                    ],  # All workers need time values
                    # Only get the slice of coordinates this worker needs
                    "lat_center": face_data_batch["lat_center"][local_start:local_end],
                    "lon_center": face_data_batch["lon_center"][local_start:local_end],
                    # Everyone needs all coordinates for partitioning
                    "lat_center_all": face_data_batch["lat_center_all"],
                    "lon_center_all": face_data_batch["lon_center_all"],
                    # Only get the slice of node connectivity this worker needs
                    "nv": face_data_batch["nv"][local_start:local_end],
                }

                # Only include node data if present (all workers need all nodes)
                if "lat_node" in face_data_batch and "lon_node" in face_data_batch:
                    worker_data_batch["lat_node"] = face_data_batch["lat_node"]
                    worker_data_batch["lon_node"] = face_data_batch["lon_node"]

                # For each variable, include only the data needed by this worker
                for var_name in vars_to_include:
                    # Skip variables we've already handled or don't exist
                    if (
                        var_name in ["nv", "lat_center", "lon_center"]
                        or var_name not in face_data_batch
                    ):
                        continue

                    # Handle layered variables
                    if f"{var_name}_has_layers" in face_data_batch:
                        worker_data_batch[f"{var_name}_has_layers"] = face_data_batch[
                            f"{var_name}_has_layers"
                        ]
                        worker_data_batch[f"{var_name}_num_layers"] = face_data_batch[
                            f"{var_name}_num_layers"
                        ]

                        for layer_idx in range(
                            face_data_batch[f"{var_name}_num_layers"]
                        ):
                            layer_key = f"{var_name}_layer_{layer_idx}"
                            if layer_key in face_data_batch:
                                # Only get the data slice this worker needs for this layer
                                try:
                                    data_slice = face_data_batch[layer_key][
                                        :, local_start:local_end
                                    ]
                                    worker_data_batch[layer_key] = data_slice
                                except (IndexError, ValueError) as e:
                                    print(f"Warning: Error slicing {layer_key}: {e}")
                                    # If slicing fails, include the whole data
                                    worker_data_batch[layer_key] = face_data_batch[
                                        layer_key
                                    ]

                    # Handle regular 2D variables
                    elif var_name in face_data_batch:
                        # Only get the data slice this worker needs
                        try:
                            if (
                                isinstance(face_data_batch[var_name], np.ndarray)
                                and len(face_data_batch[var_name].shape) > 1
                            ):
                                data_slice = face_data_batch[var_name][
                                    :, local_start:local_end
                                ]
                                worker_data_batch[var_name] = data_slice
                            else:
                                # If it's not a 2D array, just copy it (probably metadata)
                                worker_data_batch[var_name] = face_data_batch[var_name]
                        except (IndexError, ValueError) as e:
                            print(f"Warning: Error slicing {var_name}: {e}")
                            # If slicing fails, include the whole data
                            worker_data_batch[var_name] = face_data_batch[var_name]

                # Create the worker batch with this optimized data subset
                worker_batches.append(
                    (
                        worker_data_batch,  # Custom optimized subset for this worker
                        sub_batch_indices,  # Actual face indices this worker should process
                        vars_to_include,
                        attributes,
                        output_dir_str,  # Use string instead of Path
                        write_batch_size,
                    )
                )

            # Process worker batches in parallel with improved concurrency
            print(f"Starting {len(worker_batches)} worker processes in parallel...")
            worker_start_time = time.time()

            with concurrent.futures.ProcessPoolExecutor(
                max_workers=num_workers
            ) as executor:
                # Submit all jobs immediately using submit() instead of map()
                futures = []
                for worker_idx, batch in enumerate(worker_batches):
                    # Submit each job and keep track of its future
                    future = executor.submit(
                        standalone_save_parquet_batch, batch, safe_config, safe_location
                    )
                    futures.append((worker_idx, future))

                # Process results as they complete using as_completed()
                total_workers = len(futures)
                completed = 0

                for completed_future in concurrent.futures.as_completed(
                    [f[1] for f in futures]
                ):
                    try:
                        # Get result and update statistics
                        result = completed_future.result()
                        combined_stats["files_created"] += result["files_created"]
                        combined_stats["partitions_created"].update(
                            result["partitions_created"]
                        )

                        # Update completion counter and provide progress updates
                        completed += 1
                        if completed % 10 == 0 or completed == total_workers:
                            print(
                                f"Completed {completed}/{total_workers} worker processes"
                            )

                    except Exception as exc:
                        print(f"Worker process failed with error: {exc}")

            print(
                f"All worker processes completed in {time.time() - worker_start_time:.2f} seconds"
            )

            # Report progress for this main batch
            progress = int((batch_idx + 1) / num_batches * 100)
            elapsed_time = time.time() - start_time
            faces_processed = batch_end
            estimated_total = elapsed_time / faces_processed * num_faces
            remaining_time = estimated_total - elapsed_time

            print(f"Main batch {batch_idx + 1}/{num_batches} complete")
            print(
                f"Overall progress: {progress}% ({faces_processed}/{num_faces} faces)"
            )
            print(
                f"Elapsed time: {elapsed_time:.2f}s, Estimated remaining: {remaining_time:.2f}s"
            )

        # Close the dataset
        dataset.close()

        # Convert set to list for final statistics
        combined_stats["partitions_created"] = list(
            combined_stats["partitions_created"]
        )

        # Final summary report
        end_time = time.time()
        elapsed_time = end_time - start_time

        print("\nCONVERSION COMPLETED:")
        print(f"- Processed {num_faces} faces")
        print(f"- Created {combined_stats['files_created']} files")
        print(f"- Created {len(combined_stats['partitions_created'])} partitions")
        print(f"- Total time: {elapsed_time:.2f} seconds")
        print(
            f"- Average processing time per face: {elapsed_time/num_faces:.4f} seconds"
        )

        return combined_stats


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

    # Initialize the optimized converter
    converter = ConvertTidalNcToParquet(
        output_path, config, location, max_workers=max_workers
    )

    # Process each NetCDF file
    for nc_file in sorted(list(input_path.rglob("*.nc"))):
        print(f"\n{'='*80}\nProcessing file: {nc_file}\n{'='*80}")

        # Use the optimized method with increased main_batch_size and reduced write_batch_size
        stats = converter.convert_dataset(
            dataset_path=nc_file,
            write_batch_size=16,
            main_batch_size=20000,
        )

        print(f"\nFile processed: {nc_file}")
        print(f"Statistics: {stats}")

    final_output_path = file_manager.get_vap_partition_output_dir(
        config, location, use_temp_base_path=False
    )

    if output_path != final_output_path:
        print(f"Copying output files from {output_path} to {final_output_path}...")

        copy_manager.copy_directory(output_path, final_output_path)

        print(f"Copy complete! Output files are in {final_output_path}")
