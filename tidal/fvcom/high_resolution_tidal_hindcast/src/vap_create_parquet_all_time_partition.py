import concurrent.futures
import re
import json

from pathlib import Path
from typing import List, Dict, Union, Optional, Tuple, Callable

import numpy as np
import xarray as xr
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from . import copy_manager, file_manager, file_name_convention_manager, nc_manager


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
        self.output_path_map: dict[int, Path] = {}

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

    def _extract_face_data(
        self, dataset: xr.Dataset, face_indices: List[int], vars_to_include: List[str]
    ) -> Dict[int, pd.DataFrame]:
        """
        Unified method to extract data for one or multiple faces.
        Replaces _create_time_series_df and _create_time_series_dfs_batch
        """
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

        # Pre-fetch all variable data
        for var_name in vars_to_include:
            if var_name in vars_to_skip:
                continue

            var = dataset[var_name]

            # Extract data based on variable dimensions
            if "sigma_layer" in var.dims and "face" in var.dims and "time" in var.dims:
                # 3D variables (time, sigma_layer, face)
                print(f"Extracting 4D variable {var_name} with dims {var.dims}")
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
                print(f"Extracting 3D variable {var_name} with dims {var.dims}")
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

        # Create DataFrames for each face
        face_dataframes = {}
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

    def _get_face_index_string(self, filename):
        """Extract face index from filename."""
        parts = str(filename).split(".")
        for part in parts:
            if "face=" in part:
                return part.replace("face=", "")
        return None

    def _save_dataframe_to_parquet(
        self, df: pd.DataFrame, attributes: Dict, lat: float, lon: float, face_idx: int
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
        face_indices: List[int],
        vars_to_include: List[str],
        attributes: Dict,
    ) -> Dict:
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
        vars_to_include: Optional[List[str]] = None,
        max_faces: Optional[int] = None,
        batch_size: int = 1000,
        parallel: bool = False,
        write_batch_size: int = 64,
    ) -> Dict:
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
        face_indices: List[int],
        vars_to_include: List[str],
        attributes: Dict,
        write_batch_size: int,
    ) -> Dict:
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
    """
    location = config["location_specification"][location_key]
    input_path = file_manager.get_vap_output_dir(config, location)
    output_path = file_manager.get_vap_partition_output_dir(
        config, location, use_temp_base_path=True
    )

    # Initialize the converter
    converter = ConvertTidalNcToParquet(output_path, config, location)

    # Process each NetCDF file
    for nc_file in sorted(list(input_path.rglob("*.nc"))):
        print(f"Processing file: {nc_file}")
        ds = nc_manager.nc_open(nc_file, config)

        # Use the unified conversion method with parallel processing
        converter.convert_dataset(
            dataset=ds, batch_size=100000, parallel=True, write_batch_size=64
        )
    final_output_path = file_manager.get_vap_partition_output_dir(
        config, location, use_temp_base_path=False
    )

    if output_path != final_output_path:
        print(f"Copying output files from {output_path} to {final_output_path}...")

        copy_manager.copy_directory(output_path, final_output_path)

        print(f"Copy complete! Output files are in {final_output_path}")
