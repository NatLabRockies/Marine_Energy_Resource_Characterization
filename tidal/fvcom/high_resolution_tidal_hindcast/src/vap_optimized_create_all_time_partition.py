import concurrent.futures
import os
import shutil
import time
import gc
import psutil
from pathlib import Path
import numpy as np
import xarray as xr
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from . import file_manager, file_name_convention_manager, nc_manager


class OptimizedTidalConverter:
    def __init__(self, original_converter):
        self.converter = original_converter
        self.output_dir = original_converter.output_dir
        self.config = original_converter.config
        self.location = original_converter.location
        self.memory_info = self._get_system_memory()
        self.num_cores = os.cpu_count()
        self.using_nvme = "TMPDIR" in os.environ
        # Flag to control verbose logging globally
        self.verbose_logging = False
        # Track if we already did memory estimation
        self.memory_estimation_done = False

    def _get_system_memory(self):
        mem = psutil.virtual_memory()
        return {
            "total": mem.total,
            "available": mem.available,
            "percent": mem.percent,
            "safe_limit": mem.total * 0.85,  # Increased to 85% of total memory
        }

    def _estimate_face_memory_usage(self, dataset, sample_size=25):
        """Estimate average memory usage per face with a larger sample size"""
        # Enable verbose logging during memory estimation
        self.verbose_logging = True

        face_indices = np.random.choice(
            len(dataset.face.values),
            min(sample_size, len(dataset.face.values)),
            replace=False,
        )

        # Run garbage collection before measurement to get cleaner baseline
        gc.collect()
        start_mem = psutil.Process().memory_info().rss
        face_dfs = self.converter._extract_face_data(
            dataset, face_indices.tolist(), list(dataset.variables.keys())
        )
        end_mem = psutil.Process().memory_info().rss

        # Add 10% buffer to the estimate for safety
        avg_face_size = ((end_mem - start_mem) / len(face_indices)) * 1.1

        # Disable verbose logging after memory estimation
        self.verbose_logging = False
        self.memory_estimation_done = True

        return avg_face_size

    def _optimize_batch_size(self, dataset):
        """Calculate optimal batch size and worker counts based on system resources"""
        # Get memory usage estimate with better precision
        avg_face_mem = self._estimate_face_memory_usage(dataset)
        total_faces = len(dataset.face.values)

        # More aggressive memory utilization (70% of safe limit instead of 50%)
        available_mem = self.memory_info["safe_limit"] * 0.7

        # Calculate batch size based on available memory
        max_faces_per_batch = int(available_mem / avg_face_mem)

        # Apply reasonable limits to batch size
        if self.num_cores >= 64:  # For high-core systems
            max_faces_per_batch = min(
                max_faces_per_batch, 150000
            )  # Higher limit for large systems
        else:
            max_faces_per_batch = min(
                max_faces_per_batch, 75000
            )  # Moderate limit for smaller systems

        max_faces_per_batch = max(max_faces_per_batch, 1000)  # Minimum batch size

        # Worker counts scale with core count, with reasonable limits
        if self.num_cores >= 64:  # For systems with many cores like Kestrel
            # More aggressive worker counts for high-core systems
            cpu_workers = min(
                max(1, int(self.num_cores * 0.75)), 64
            )  # Use 75% of cores, max 64
            io_workers = min(
                max(1, int(self.num_cores * 0.5)), 96
            )  # Use 50% of cores, max 96
        else:
            # More conservative for smaller systems
            cpu_workers = min(
                max(1, int(self.num_cores * 0.6)), 32
            )  # Use 60% of cores, max 32
            io_workers = min(
                max(1, int(self.num_cores * 0.4)), 48
            )  # Use 40% of cores, max 48

        return {
            "batch_size": max_faces_per_batch,
            "cpu_workers": cpu_workers,
            "io_workers": io_workers,
            "estimated_mem_per_face": avg_face_mem,
            "total_faces": total_faces,
        }

    def _setup_local_storage(self):
        if self.using_nvme:
            tmp_dir = os.environ.get("TMPDIR")
            temp_output = Path(tmp_dir) / f"temp_parquet_{os.getpid()}"
        else:
            temp_output = Path("/tmp") / f"temp_parquet_{os.getpid()}"

        temp_output.mkdir(exist_ok=True, parents=True)
        return temp_output

    def _process_chunk(self, chunk_faces, dataset, vars_to_include, temp_converter):
        """Separate method for processing a chunk of faces to avoid pickling issues"""
        # Ensure we're using the correct verbose setting in subprocesses
        if hasattr(self, "verbose_logging"):
            # If we have a memory_estimation_done attribute, use that to determine verbosity
            if hasattr(self, "memory_estimation_done") and self.memory_estimation_done:
                # We're past memory estimation, force quiet mode
                temp_verbose = False
            else:
                temp_verbose = self.verbose_logging

            # Transfer the verbose setting to the converter if it has that attribute
            if hasattr(temp_converter, "verbose_logging"):
                temp_converter.verbose_logging = temp_verbose

        return temp_converter._extract_face_data(dataset, chunk_faces, vars_to_include)

    def _process_in_chunks(
        self,
        dataset,
        face_indices,
        vars_to_include,
        temp_converter,
        attributes,
        cpu_workers,
    ):
        chunk_size = max(1, len(face_indices) // cpu_workers)
        chunks = [
            face_indices[i : i + chunk_size]
            for i in range(0, len(face_indices), chunk_size)
        ]

        all_face_dfs = {}

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=cpu_workers
        ) as executor:
            # Using a class method instead of a local function to avoid pickling issues
            futures = []
            for chunk in chunks:
                future = executor.submit(
                    self._process_chunk, chunk, dataset, vars_to_include, temp_converter
                )
                futures.append(future)

            # Get results as they complete
            for future in concurrent.futures.as_completed(futures):
                chunk_result = future.result()
                all_face_dfs.update(chunk_result)

        return all_face_dfs

    def _write_task(self, args):
        """Separate method for writing a task to avoid pickling issues"""
        df, attrs, lat, lon, face_idx = args
        return self.converter._save_dataframe_to_parquet(df, attrs, lat, lon, face_idx)

    def _parallel_write(
        self, all_face_dfs, dataset, temp_converter, attributes, io_workers
    ):
        lat_center = dataset.lat_center.values
        lon_center = dataset.lon_center.values

        write_tasks = []
        for face_idx, df in all_face_dfs.items():
            lat = float(lat_center[face_idx])
            lon = float(lon_center[face_idx])
            write_tasks.append((df, attributes, lat, lon, face_idx))

        with concurrent.futures.ThreadPoolExecutor(max_workers=io_workers) as executor:
            futures = []
            for task in write_tasks:
                future = executor.submit(self._write_task, task)
                futures.append(future)

            saved_paths = []
            for future in concurrent.futures.as_completed(futures):
                saved_paths.append(future.result())

        return saved_paths

    def _copy_results(self, temp_dir):
        """Copy results from temporary storage to final destination with proper error handling"""
        final_dir = self.converter.output_dir

        try:
            # Get list of all parquet files
            all_files = list(temp_dir.glob("**/*.parquet"))
            total_files = len(all_files)

            if total_files == 0:
                print("Warning: No parquet files found in temporary directory")
                return

            copied = 0

            for src_path in all_files:
                try:
                    # Extract partition pattern (lat_deg=XX/lon_deg=XX/etc)
                    partition_pattern = None
                    path_str = str(src_path)
                    if "lat_deg=" in path_str:
                        # Find the partition pattern in the path
                        idx = path_str.find("lat_deg=")
                        end_idx = path_str.rfind("/")
                        if end_idx > idx:
                            partition_pattern = path_str[idx:end_idx]
                        else:
                            partition_pattern = path_str[idx:]

                    if partition_pattern:
                        # Construct destination path using extracted pattern
                        dst_path = final_dir / partition_pattern / src_path.name
                    else:
                        # Fallback: try to preserve directory structure relative to temp_dir
                        try:
                            rel_path = src_path.relative_to(temp_dir)
                            dst_path = final_dir / rel_path
                        except ValueError:
                            # Last resort: just use the filename
                            dst_path = final_dir / src_path.name

                    # Ensure destination directory exists
                    dst_path.parent.mkdir(exist_ok=True, parents=True)

                    # Copy the file
                    shutil.copy2(src_path, dst_path)

                    copied += 1
                    if copied % 1000 == 0 or copied == total_files:
                        print(
                            f"Copied {copied}/{total_files} files ({(copied/total_files)*100:.1f}%)"
                        )

                except Exception as e:
                    print(
                        f"Warning: Error copying {src_path} to final destination: {e}"
                    )

        except Exception as e:
            print(f"Error during copy operation: {e}")
            # Continue with what we have to avoid losing all processed data

    def convert_file(self, nc_file):
        temp_dir = self._setup_local_storage()
        temp_converter = type(self.converter)(
            temp_dir, self.converter.config, self.converter.location
        )

        ds = nc_manager.nc_open(nc_file, self.converter.config)
        attributes = temp_converter._extract_attributes(ds)
        vars_to_include = list(ds.variables.keys())

        optimization = self._optimize_batch_size(ds)
        batch_size = optimization["batch_size"]
        cpu_workers = optimization["cpu_workers"]
        io_workers = optimization["io_workers"]

        print(
            f"Optimized parameters: batch_size={batch_size}, cpu_workers={cpu_workers}, io_workers={io_workers}"
        )
        print(
            f"Estimated memory per face: {optimization['estimated_mem_per_face'] / (1024*1024):.2f} MB"
        )

        num_faces = optimization["total_faces"]
        total_batches = (num_faces + batch_size - 1) // batch_size

        stats = {
            "total_faces": num_faces,
            "partitions_created": set(),
            "files_created": 0,
        }

        for batch_idx, batch_start in enumerate(range(0, num_faces, batch_size)):
            batch_end = min(batch_start + batch_size, num_faces)
            face_indices = list(range(batch_start, batch_end))

            print(
                f"Processing batch {batch_idx+1}/{total_batches}: faces {batch_start}-{batch_end-1}"
            )

            start_time = time.time()
            all_face_dfs = self._process_in_chunks(
                ds,
                face_indices,
                vars_to_include,
                temp_converter,
                attributes,
                cpu_workers,
            )
            process_time = time.time() - start_time

            write_start = time.time()
            saved_paths = self._parallel_write(
                all_face_dfs, ds, temp_converter, attributes, io_workers
            )
            write_time = time.time() - write_start

            stats["files_created"] += len(saved_paths)

            # Handle paths correctly to avoid relative_to errors
            for path in saved_paths:
                path_obj = Path(path)
                if path_obj.exists():
                    if self.using_nvme:
                        # For temporary storage, we track only the partition pattern
                        # Extract the lat/lon partition pattern (lat_deg=XX/lon_deg=XX/etc)
                        try:
                            partition_pattern = str(path_obj.parent)
                            # Find the lat_deg part in the path
                            if "lat_deg=" in partition_pattern:
                                # Extract just the partition pattern
                                idx = partition_pattern.find("lat_deg=")
                                stats["partitions_created"].add(partition_pattern[idx:])
                        except Exception as e:
                            print(
                                f"Warning: Could not extract partition pattern from {path_obj}: {e}"
                            )
                    else:
                        # For direct storage, use relative path
                        try:
                            rel_path = path_obj.parent.relative_to(
                                self.converter.output_dir
                            )
                            stats["partitions_created"].add(str(rel_path))
                        except ValueError:
                            # If relative_to fails, just store the full path
                            stats["partitions_created"].add(str(path_obj.parent))

            del all_face_dfs
            gc.collect()

            current_mem = psutil.Process().memory_info().rss / (1024 * 1024 * 1024)
            progress = (batch_end / num_faces) * 100

            print(
                f"Batch {batch_idx+1} complete. Files: {len(saved_paths)}, "
                f"Process time: {process_time:.2f}s, Write time: {write_time:.2f}s, "
                f"Current memory: {current_mem:.2f} GB, Progress: {progress:.1f}%"
            )

        if self.using_nvme:
            print("Copying results from temporary storage to final destination")
            self._copy_results(temp_dir)
            shutil.rmtree(temp_dir)

        # Ensure partitions_created is a list of strings for serialization
        stats["partitions_created"] = [str(p) for p in stats["partitions_created"]]

        print(f"Conversion complete for {nc_file}")
        return stats

        for batch_idx, batch_start in enumerate(range(0, num_faces, batch_size)):
            batch_end = min(batch_start + batch_size, num_faces)
            face_indices = list(range(batch_start, batch_end))

            print(
                f"Processing batch {batch_idx+1}/{total_batches}: faces {batch_start}-{batch_end-1}"
            )

            start_time = time.time()
            all_face_dfs = self._process_in_chunks(
                ds,
                face_indices,
                vars_to_include,
                temp_converter,
                attributes,
                cpu_workers,
            )
            process_time = time.time() - start_time

            write_start = time.time()
            saved_paths = self._parallel_write(
                all_face_dfs, ds, temp_converter, attributes, io_workers
            )
            write_time = time.time() - write_start

            stats["files_created"] += len(saved_paths)
            for path in saved_paths:
                path_obj = Path(path)
                if path_obj.exists():
                    stats["partitions_created"].add(
                        path_obj.parent.relative_to(temp_dir)
                    )

            del all_face_dfs
            gc.collect()

            current_mem = psutil.Process().memory_info().rss / (1024 * 1024 * 1024)
            progress = (batch_end / num_faces) * 100

            print(
                f"Batch {batch_idx+1} complete. Files: {len(saved_paths)}, "
                f"Process time: {process_time:.2f}s, Write time: {write_time:.2f}s, "
                f"Current memory: {current_mem:.2f} GB, Progress: {progress:.1f}%"
            )

        if self.using_nvme:
            print("Copying results from temporary storage to final destination")
            self._copy_results(temp_dir)
            shutil.rmtree(temp_dir)

        # Convert set to list for serialization
        stats["partitions_created"] = list(stats["partitions_created"])

        print(f"Conversion complete for {nc_file}")
        return stats


class ConvertTidalNcToParquet:
    """
    Converts an xarray Dataset with FVCOM structure to partitioned Parquet files.
    Each face in the dataset will be converted to a time-indexed Parquet file
    stored in a partition based on its lat/lon coordinates.
    """

    def __init__(self, output_dir: str, config=None, location=None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.config = config
        self.location = location
        self.output_path_map = {}

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
    def _extract_attributes(dataset: xr.Dataset) -> dict:
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

    def _extract_face_data(
        self, dataset: xr.Dataset, face_indices: list, vars_to_include: list
    ) -> dict:
        """
        Unified method to extract data for one or multiple faces with cleaner progress reporting.
        """
        verbose_mode = self.verbose_logging

        if verbose_mode:
            print(f"Processing {len(face_indices)} faces")

        # Common variables needed for all faces
        time_values = dataset.time.values
        time_dim_len = len(time_values)

        # Pre-fetch static data for all faces
        batch_data = {
            "lat_center": dataset.lat_center.values[face_indices],
            "lon_center": dataset.lon_center.values[face_indices],
            "nv": dataset["nv"].isel(time=0).isel(face=face_indices).values.T,
        }

        # Pre-fetch lat_node and lon_node if they exist
        if "lat_node" in dataset.variables and "lon_node" in dataset.variables:
            batch_data["lat_node"] = dataset["lat_node"].values
            batch_data["lon_node"] = dataset["lon_node"].values

        # Variables to skip in extraction
        vars_to_skip = ["nv", "h_center"]

        # Track progress for variables
        total_vars = sum(1 for var in vars_to_include if var not in vars_to_skip)
        processed_vars = 0

        # Pre-fetch all variable data
        for var_name in vars_to_include:
            if var_name in vars_to_skip:
                continue

            var = dataset[var_name]
            processed_vars += 1

            # Only show detailed variable logging in verbose mode
            if verbose_mode:
                if (
                    "sigma_layer" in var.dims
                    and "face" in var.dims
                    and "time" in var.dims
                ):
                    print(f"Extracting 4D variable {var_name} with dims {var.dims}")
                elif "face" in var.dims and "time" in var.dims:
                    print(f"Extracting 3D variable {var_name} with dims {var.dims}")
            elif processed_vars % 5 == 0 or processed_vars == total_vars:
                # Show simplified progress update every 5 variables
                print(
                    f"\rExtracting variables: {processed_vars}/{total_vars} ({(processed_vars/total_vars)*100:.0f}%)...",
                    end="" if processed_vars < total_vars else "\n",
                )

            # Extract data based on variable dimensions
            if "sigma_layer" in var.dims and "face" in var.dims and "time" in var.dims:
                # 3D variables (time, sigma_layer, face)
                selected_data = var.isel(face=face_indices)

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

        # Create DataFrames with progress reporting
        face_dataframes = {}
        total_faces = len(face_indices)

        print(f"Creating DataFrames for {total_faces} faces...")
        progress_interval = max(
            1, min(total_faces // 10, 1000)
        )  # Update every 10% or every 1000 faces

        for i, face_idx in enumerate(face_indices):
            # Show progress updates
            if (i + 1) % progress_interval == 0 or i + 1 == total_faces:
                progress = ((i + 1) / total_faces) * 100
                print(
                    f"\rCreating DataFrames: {i+1}/{total_faces} ({progress:.1f}%)...",
                    end="" if i + 1 < total_faces else "\n",
                )

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

                    data_dict[f"element_corner_{corner_num}_lat"] = np.repeat(
                        lat_node_val, time_dim_len
                    )
                    data_dict[f"element_corner_{corner_num}_lon"] = np.repeat(
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
                            if verbose_mode:
                                print(
                                    f"Warning: {col_name} for face {face_idx} has time dimension {len(var_data)} != {time_dim_len}"
                                )
                            continue

                        data_dict[col_name] = var_data

                elif (
                    "face" in dataset[var_name].dims
                    and "time" in dataset[var_name].dims
                ):
                    # Handle 2D variables
                    var_data = batch_data[var_name][:, i]

                    # Verify data length
                    if len(var_data) != time_dim_len:
                        if verbose_mode:
                            print(
                                f"Warning: {var_name} for face {face_idx} has time dimension {len(var_data)} != {time_dim_len}"
                            )
                        continue

                    data_dict[var_name] = var_data

            # Create DataFrame with time index
            df = pd.DataFrame(data_dict, index=time_values)
            df.index.name = "time"
            face_dataframes[face_idx] = df

        return face_dataframes

    @staticmethod
    def _prepare_netcdf_compatible_metadata(attributes: dict) -> dict:
        """
        Process and prepare metadata to be compatible with NetCDF/xarray structure.
        """
        import json

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

    def _get_face_index_string(self, filename):
        """Extract face index from filename."""
        parts = str(filename).split(".")
        for part in parts:
            if "face=" in part:
                return part.replace("face=", "")
        return None

    def _save_dataframe_to_parquet(
        self, df: pd.DataFrame, attributes: dict, lat: float, lon: float, face_idx: int
    ) -> str:
        """
        Unified method to save DataFrame to Parquet with proper partitioning and metadata.
        """
        # Generate partition path and create directory
        partition_path = self._get_partition_path(lat, lon)
        full_dir = Path(self.output_dir, partition_path)
        full_dir.mkdir(exist_ok=True, parents=True)

        # Generate filename
        filename = f"face_{face_idx}.parquet"
        if self.config is not None:
            filename = self._get_partition_file_name(face_idx, lat, lon, df)

        # Check for existing files with same face index
        output_filename = filename
        if face_idx is not None:
            existing_files = sorted(list(full_dir.glob("*.parquet")))
            matching_files = [
                file for file in existing_files if f"face={face_idx}" in file.name
            ]

            # Handle existing files
            if matching_files:
                output_filename = matching_files[0].name
                # Concatenate all existing files with new data
                dfs = [df]

                for file in matching_files:
                    existing_df = pd.read_parquet(file)
                    dfs.append(existing_df)
                    file.unlink()  # Delete after reading

                # Combine, deduplicate, and sort
                df = pd.concat(dfs).sort_index()

        # Full path to output file
        full_path = Path(full_dir, output_filename)

        # Convert to PyArrow table with metadata
        table = pa.Table.from_pandas(df)
        metadata_bytes = self._prepare_netcdf_compatible_metadata(attributes)
        table = table.replace_schema_metadata(
            {**table.schema.metadata, **metadata_bytes}
        )

        # Write to parquet
        pq.write_table(table, full_path)
        return str(full_path)

    def _process_batch(
        self,
        dataset: xr.Dataset,
        face_indices: list,
        vars_to_include: list,
        attributes: dict,
    ) -> dict:
        """
        Process a batch of faces - extract data and save to Parquet files.
        """
        # Extract data for all faces in batch
        face_dfs = self._extract_face_data(dataset, face_indices, vars_to_include)

        # Get coordinates for all faces
        lat_center = dataset.lat_center.values
        lon_center = dataset.lon_center.values

        # Prepare write tasks
        write_tasks = []
        for face_idx, df in face_dfs.items():
            lat = float(lat_center[face_idx])
            lon = float(lon_center[face_idx])
            write_tasks.append((df, attributes, lat, lon, face_idx))

        # Process write tasks
        saved_files = []
        for df, attrs, lat, lon, face_idx in write_tasks:
            saved_path = self._save_dataframe_to_parquet(df, attrs, lat, lon, face_idx)
            saved_files.append(saved_path)

        # Return statistics for this batch
        return {
            "files_created": len(saved_files),
            "partitions_created": set(
                Path(p).parent.relative_to(self.output_dir) for p in saved_files
            ),
        }

    def convert_dataset(
        self,
        dataset: xr.Dataset,
        vars_to_include: list = None,
        max_faces: int = None,
        batch_size: int = 1000,
        parallel: bool = False,
        write_batch_size: int = 64,
    ) -> dict:
        """
        Unified method to convert an xarray Dataset to partitioned Parquet files.
        Combines functionality of convert_dataset, convert_dataset_batched, and convert_dataset_parallel

        Parameters
        ----------
        dataset : xr.Dataset
            The input dataset
        vars_to_include : List[str], optional
            List of variable names to include. If None, includes all variables.
        max_faces : int, optional
            Maximum number of faces to process
        batch_size : int, optional
            Number of faces to process in each batch
        parallel : bool, optional
            Whether to use parallel processing
        write_batch_size : int, optional
            Number of files to write in parallel (only used if parallel=True)

        Returns
        -------
        Dict
            Statistics about the conversion process
        """
        # Extract attributes and determine variables to include
        attributes = self._extract_attributes(dataset)
        if vars_to_include is None:
            vars_to_include = list(dataset.variables.keys())

        # Determine number of faces to process
        num_faces = len(dataset.lat_center.values)
        if max_faces is not None:
            num_faces = min(num_faces, max_faces)

        # Initialize statistics
        stats = {
            "total_faces": num_faces,
            "partitions_created": set(),
            "files_created": 0,
        }

        # Process faces in batches
        for batch_start in range(0, num_faces, batch_size):
            batch_end = min(batch_start + batch_size, num_faces)
            face_indices = list(range(batch_start, batch_end))

            print(
                f"Processing batch of faces {batch_start} to {batch_end-1} (batch size: {len(face_indices)})"
            )

            if parallel:
                # Use parallel processing for this batch
                batch_stats = self._process_batch_parallel(
                    dataset, face_indices, vars_to_include, attributes, write_batch_size
                )
            else:
                # Use sequential processing for this batch
                batch_stats = self._process_batch(
                    dataset, face_indices, vars_to_include, attributes
                )

            # Update overall statistics
            stats["files_created"] += batch_stats["files_created"]
            stats["partitions_created"].update(batch_stats["partitions_created"])

            # Report progress
            progress = int((batch_end / num_faces) * 100)
            print(f"Progress: {progress}% ({batch_end}/{num_faces} faces processed)")

        # Convert set to list for serialization
        stats["partitions_created"] = list(stats["partitions_created"])
        return stats

    def _process_batch_parallel(
        self,
        dataset: xr.Dataset,
        face_indices: list,
        vars_to_include: list,
        attributes: dict,
        write_batch_size: int,
    ) -> dict:
        """
        Process a batch of faces using parallel I/O operations.
        """
        # Extract data for all faces in batch
        face_dfs = self._extract_face_data(dataset, face_indices, vars_to_include)

        # Get coordinates for all faces
        lat_center = dataset.lat_center.values
        lon_center = dataset.lon_center.values

        # Prepare write tasks
        all_write_tasks = []
        for face_idx, df in face_dfs.items():
            lat = float(lat_center[face_idx])
            lon = float(lon_center[face_idx])

            # Generate partition path and create directory
            partition_path = self._get_partition_path(lat, lon)
            full_dir = Path(self.output_dir, partition_path)
            full_dir.mkdir(exist_ok=True, parents=True)

            # Generate filename
            filename = f"face_{face_idx}.parquet"
            if self.config is not None:
                filename = self._get_partition_file_name(face_idx, lat, lon, df)

            all_write_tasks.append((df, attributes, partition_path, filename, face_idx))

        # Process write tasks in parallel batches
        saved_paths = []
        partitions_created = set()

        for i in range(0, len(all_write_tasks), write_batch_size):
            batch_tasks = all_write_tasks[i : i + write_batch_size]
            batch_paths = self._save_parquet_with_metadata_parallel(batch_tasks)
            saved_paths.extend(batch_paths)

            # Track partitions created
            for path_str in batch_paths:
                path = Path(path_str)
                partitions_created.add(path.parent.relative_to(self.output_dir))

        return {
            "files_created": len(saved_paths),
            "partitions_created": partitions_created,
        }

    @staticmethod
    def _parallel_save_parquet(args):
        """Helper function to save a DataFrame to Parquet in parallel."""
        table, full_path = args
        pq.write_table(table, full_path)
        return str(full_path)

    def _save_parquet_with_metadata_parallel(self, write_tasks):
        """Prepare and save multiple DataFrames to Parquet with metadata in parallel."""
        batch_tasks = []

        for df, attributes, partition_path, filename, face_idx in write_tasks:
            # Handle existing files
            if face_idx in self.output_path_map:
                # Concat with existing file
                full_path = self.output_path_map[face_idx]
                existing_df = pd.read_parquet(full_path)
                output_df = pd.concat([existing_df, df]).sort_index()
                output_path = full_path
            else:
                # Create new file
                full_dir = Path(self.output_dir, partition_path)
                full_dir.mkdir(exist_ok=True, parents=True)
                full_path = Path(full_dir, filename)
                self.output_path_map[face_idx] = full_path
                output_df = df
                output_path = full_path

            # Convert to PyArrow table with metadata
            table = pa.Table.from_pandas(output_df)
            metadata_bytes = self._prepare_netcdf_compatible_metadata(attributes)
            table = table.replace_schema_metadata(
                {**table.schema.metadata, **metadata_bytes}
            )
            batch_tasks.append((table, output_path))

        # Execute parallel writes
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(self._parallel_save_parquet, batch_tasks))
        return results


def partition_vap_into_parquet_dataset(config, location_key):
    """
    Process VAP data and convert to partitioned Parquet files.

    This is the entry point function with the same API as the original code,
    but using optimized implementation for better memory management and parallel processing.
    """
    import time
    from datetime import timedelta

    # Start total execution timer
    total_start_time = time.time()

    location = config["location_specification"][location_key]
    input_path = file_manager.get_vap_output_dir(config, location)
    output_path = file_manager.get_vap_partition_output_dir(config, location)

    # Initialize the original converter
    original_converter = ConvertTidalNcToParquet(output_path, config, location)

    # Create optimized wrapper
    optimizer = OptimizedTidalConverter(original_converter)

    # Track overall statistics
    overall_stats = {
        "total_files_created": 0,
        "total_partitions_created": set(),
        "files_processed": 0,
        "total_files": len(list(input_path.rglob("*.nc"))),
        "timing": {
            "total_elapsed": 0,
            "files": {},
            "avg_time_per_file": 0,
            "avg_time_per_face": 0,
            "total_faces_processed": 0,
        },
    }

    # Process each NetCDF file
    for nc_file in sorted(list(input_path.rglob("*.nc"))):
        file_start_time = time.time()
        print(f"Processing file: {nc_file}")

        # Convert file using optimized approach
        file_stats = optimizer.convert_file(nc_file)

        # Calculate file processing time
        file_elapsed_time = time.time() - file_start_time
        file_elapsed_str = str(timedelta(seconds=int(file_elapsed_time)))

        # Update timing statistics
        overall_stats["timing"]["files"][str(nc_file)] = {
            "elapsed_seconds": file_elapsed_time,
            "elapsed_formatted": file_elapsed_str,
            "faces_processed": file_stats["total_faces"],
            "seconds_per_face": file_elapsed_time / max(1, file_stats["total_faces"]),
        }
        overall_stats["timing"]["total_faces_processed"] += file_stats["total_faces"]

        # Update overall statistics
        overall_stats["total_files_created"] += file_stats["files_created"]
        if isinstance(file_stats["partitions_created"], list):
            overall_stats["total_partitions_created"].update(
                file_stats["partitions_created"]
            )
        else:
            overall_stats["total_partitions_created"].update(
                list(file_stats["partitions_created"])
            )
        overall_stats["files_processed"] += 1

        # Report progress with timing
        progress = (
            overall_stats["files_processed"] / overall_stats["total_files"]
        ) * 100
        total_elapsed_so_far = time.time() - total_start_time
        avg_time_per_file = total_elapsed_so_far / overall_stats["files_processed"]
        estimated_time_remaining = avg_time_per_file * (
            overall_stats["total_files"] - overall_stats["files_processed"]
        )

        print(f"File completed in {file_elapsed_str}")
        print(
            f"Overall progress: {progress:.1f}% ({overall_stats['files_processed']}/{overall_stats['total_files']} files)"
        )
        print(f"Elapsed time: {str(timedelta(seconds=int(total_elapsed_so_far)))}")
        print(
            f"Estimated time remaining: {str(timedelta(seconds=int(estimated_time_remaining)))}"
        )

    # Calculate final timing statistics
    total_elapsed = time.time() - total_start_time
    overall_stats["timing"]["total_elapsed"] = total_elapsed
    overall_stats["timing"]["total_elapsed_formatted"] = str(
        timedelta(seconds=int(total_elapsed))
    )

    # Calculate averages
    if overall_stats["files_processed"] > 0:
        overall_stats["timing"]["avg_time_per_file"] = (
            total_elapsed / overall_stats["files_processed"]
        )

    if overall_stats["timing"]["total_faces_processed"] > 0:
        overall_stats["timing"]["avg_time_per_face"] = (
            total_elapsed / overall_stats["timing"]["total_faces_processed"]
        )

    # Convert set to list for final statistics
    overall_stats["total_partitions_created"] = list(
        overall_stats["total_partitions_created"]
    )

    # Print final statistics with timing information
    print("\n" + "=" * 50)
    print("PROCESSING COMPLETE")
    print("=" * 50)
    print(f"Total elapsed time: {overall_stats['timing']['total_elapsed_formatted']}")
    print(f"Files processed: {overall_stats['files_processed']}")
    print(f"Total faces processed: {overall_stats['timing']['total_faces_processed']}")
    print(f"Files created: {overall_stats['total_files_created']}")
    print(f"Partitions created: {len(overall_stats['total_partitions_created'])}")
    print(
        f"Average time per file: {str(timedelta(seconds=int(overall_stats['timing']['avg_time_per_file'])))}"
    )
    print(
        f"Average time per face: {overall_stats['timing']['avg_time_per_face']:.4f} seconds"
    )
    print("=" * 50)

    return overall_stats
