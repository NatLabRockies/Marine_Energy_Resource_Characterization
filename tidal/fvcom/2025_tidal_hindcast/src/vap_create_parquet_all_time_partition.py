import concurrent.futures
import re
import json

from pathlib import Path
from typing import List, Dict, Union, Optional

import numpy as np
import xarray as xr
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from . import file_manager, file_name_convention_manager


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

    @staticmethod
    def _get_partition_path(lat: float, lon: float) -> str:
        """
        Generate the partition path based on lat/lon coordinates.

        Parameters
        ----------
        lat : float
            Latitude value
        lon : float
            Longitude value

        Returns
        -------
        str
            Partition path in the format: lat_deg=XX/lon_deg=YY/lat_dec=ZZ/lon_dec=WW
        """
        lat_deg = int(lat)
        lon_deg = int(lon)
        lat_dec = int(abs(lat * 100) % 100)
        lon_dec = int(abs(lon * 100) % 100)

        return (
            f"lat_deg={lat_deg}/lon_deg={lon_deg}/lat_dec={lat_dec}/lon_dec={lon_dec}"
        )

    def _get_partition_file_name(
        self,
        index: int,
        lat: float,
        lon: float,
        df,
        index_max_digits=6,
        # 1.1 cm precision
        coord_digits_max=7,
    ) -> str:
        # Round latitude and longitude to specified decimal places
        lat_rounded = round(lat, coord_digits_max)
        lon_rounded = round(lon, coord_digits_max)

        expected_delta_t_seconds = self.location["expected_delta_t_seconds"]
        if expected_delta_t_seconds == 3600:
            temporal_string = "1h"
        elif expected_delta_t_seconds == 1800:
            temporal_string = "30m"
        else:
            raise ValueError(
                f"Unexpected expected_delta_t_seconds configuration {expected_delta_t_seconds}"
            )

        # Use the parameters in the format strings
        # Create the zero-padding format for the index based on index_max_digits
        index_format = f"0{index_max_digits}d"
        # Create the decimal precision format for coordinates based on coord_digits_max
        coord_format = f".{coord_digits_max}f"

        return file_name_convention_manager.generate_filename_for_data_level(
            df,
            self.location["output_name"],
            f"{self.config['dataset']['name']}.face={index:{index_format}}.lat={lat_rounded:{coord_format}}.lon={lon_rounded:{coord_format}}",
            "b3",
            temporal=temporal_string,
        )

    @staticmethod
    def _extract_attributes(dataset: xr.Dataset) -> Dict:
        """
        Extract global and variable attributes from the dataset.

        Parameters
        ----------
        dataset : xr.Dataset
            The input dataset

        Returns
        -------
        Dict
            Dictionary containing global and variable attributes
        """
        # Get global attributes
        global_attrs = dict(dataset.attrs)

        # Get variable attributes
        var_attrs = {}
        for var_name, var in dataset.variables.items():
            var_attrs[var_name] = dict(var.attrs)

        return {"global_attrs": global_attrs, "variable_attrs": var_attrs}

    @staticmethod
    def _create_time_series_df(
        dataset: xr.Dataset, face_idx: int, vars_to_include: List[str]
    ) -> pd.DataFrame:
        """
        Create a time-indexed DataFrame for a specific face.

        Parameters
        ----------
        dataset : xr.Dataset
            The input dataset
        face_idx : int
            The face index to extract
        vars_to_include : List[str]
            List of variable names to include

        Returns
        -------
        pd.DataFrame
            DataFrame with time as index and variables as columns
        """
        # Dictionary to store data for DataFrame
        data_dict = {}

        print(f"Vars to include are: {vars_to_include}")

        # Get time values for index
        print("Extracting time values...")
        time_values = dataset.time.values

        data_dict["lat_center"] = [dataset.lat_center.values[face_idx]] * len(
            time_values
        )
        data_dict["lon_center"] = [dataset.lon_center.values[face_idx]] * len(
            time_values
        )

        # Extract triangle vertex information from nv (node vertex) variable
        # nv uses Fortran-style indexing, so we need to adjust
        # Get the first timestamp since topology doesn't change
        print("Extracting nv values...")
        nv_data = dataset["nv"].isel(time=0, face=face_idx).values

        # Get node indices for the three corners of this face
        # Adjust for possible Fortran 1-based indexing by subtracting 1
        node_indices = [int(idx - 1) if idx > 0 else int(idx) for idx in nv_data]

        print("Extracting data for each corner node...")
        # Extract lat/lon for each corner node
        if "lat_node" in dataset.variables and "lon_node" in dataset.variables:
            for i, node_idx in enumerate(node_indices):
                # Add vertex information as static columns
                corner_num = i + 1  # 1-based for clarity

                # Add corner lat/lon values
                lat_node_val = float(dataset["lat_node"].values[node_idx])
                lon_node_val = float(dataset["lon_node"].values[node_idx])

                # Add to data dict with repeated values for each time step
                data_dict[f"element_corner_{corner_num}_lat"] = np.repeat(
                    lat_node_val, len(time_values)
                )
                data_dict[f"element_corner_{corner_num}_lon"] = np.repeat(
                    lon_node_val, len(time_values)
                )

        # vars_to_include.remove("lat_node")
        # vars_to_include.remove("lon_node")
        # vars_to_include.remove("face_node_index")
        # vars_to_include.remove("time")

        print("===LOOPING===")

        vars_to_skip = ["nv", "h_center"]

        # Extract data for each variable
        for var_name in vars_to_include:
            var = dataset[var_name]

            if var_name in vars_to_skip:
                continue

            # Check variable dimensions
            if "sigma_layer" in var.dims and "face" in var.dims:
                print(f"Extracting data for {var_name}")
                # Handle 3D variables (time, sigma_layer, face)
                for layer_idx in range(len(dataset.sigma_layer)):
                    # Create column name with layer information
                    col_name = f"{var_name}_layer_{layer_idx}"
                    # Extract data for specific face and layer across all times
                    data_dict[col_name] = var.isel(
                        face=face_idx, sigma_layer=layer_idx
                    ).values
                    print(f"{col_name}.shape = {data_dict[col_name].shape}")

            elif "face" in var.dims and "time" in var.dims:
                print(f"Extracting data for {var_name}")
                # Handle 2D variables (time, face)
                data_dict[var_name] = var.isel(face=face_idx).values
                print(f"{var_name}.shape = {data_dict[var_name].shape}")

            # elif "face" in var.dims and "time" not in var.dims:
            #     # Handle static face variables (e.g., lat_center, lon_center)
            #     # Repeat the value for each time step
            #     value = var.isel(face=face_idx).values
            #     data_dict[var_name] = np.repeat(value, len(time_values))

        # Create DataFrame with time as index
        df = pd.DataFrame(data_dict, index=time_values)
        df.index.name = "time"

        return df

    @staticmethod
    def _create_time_series_dfs_batch(
        dataset: xr.Dataset, face_indices: List[int], vars_to_include: List[str]
    ) -> Dict[int, pd.DataFrame]:
        """
        Create time-indexed DataFrames for multiple faces at once with optimized batch data fetching.

        Parameters
        ----------
        dataset : xr.Dataset
            The input dataset
        face_indices : List[int]
            List of face indices to extract
        vars_to_include : List[str]
            List of variable names to include

        Returns
        -------
        Dict[int, pd.DataFrame]
            Dictionary mapping face indices to DataFrames with time as index
        """
        print(f"Processing {len(face_indices)} faces in batch")

        # Get time values for index
        time_values = dataset.time.values
        time_dim_len = len(time_values)
        print(f"Time dimension length: {time_dim_len}")

        # Dictionary to store batch data fetched from xarray
        batch_data = {}

        # Pre-fetch static data for the batch
        batch_data["lat_center"] = dataset.lat_center.values[face_indices]
        batch_data["lon_center"] = dataset.lon_center.values[face_indices]

        # Get nv data for all faces at once
        batch_data["nv"] = dataset["nv"].isel(time=0).isel(face=face_indices).values.T
        print(f"nv shape: {batch_data['nv'].shape}")
        # nv_data = batch_data["nv"][i]
        print(f"nv data[0]: {batch_data['nv'][0]}")
        print(f"nv data[1]: {batch_data['nv'][1]}")

        # Pre-fetch lat_node and lon_node if they exist
        if "lat_node" in dataset.variables and "lon_node" in dataset.variables:
            batch_data["lat_node"] = dataset["lat_node"].values
            batch_data["lon_node"] = dataset["lon_node"].values

        # Pre-fetch all variable data
        vars_to_skip = ["nv", "h_center"]

        for var_name in vars_to_include:
            if var_name in vars_to_skip:
                continue

            var = dataset[var_name]
            print(f"Processing variable {var_name} with dims {var.dims}")

            # Check variable dimensions and fetch accordingly
            if "sigma_layer" in var.dims and "face" in var.dims and "time" in var.dims:
                # 3D variables (time, sigma_layer, face)
                # Select all requested faces at once for all layers
                # This preserves all dimensions but filters to just our faces
                selected_data = var.isel(face=face_indices)

                batch_data[var_name] = {}

                for layer_idx in range(len(dataset.sigma_layer)):
                    # Now extract the specific layer from our already-filtered data
                    layer_data = selected_data.isel(sigma_layer=layer_idx)

                    # At this point, we should have dimensions [time, face]
                    # Ensure the data is in the expected shape for efficient slicing
                    if layer_data.dims[0] == "time" and layer_data.dims[1] == "face":
                        data_array = layer_data.values
                    else:
                        # Transpose to ensure [time, face] ordering
                        data_array = layer_data.transpose("time", "face").values

                    # Store the data array
                    batch_data[var_name][layer_idx] = data_array

            elif "face" in var.dims and "time" in var.dims:
                # 2D variables (time, face)
                # Select all faces at once
                faces_data = var.isel(face=face_indices)

                # Get time dimension index (assume 0, but check)
                time_dim_idx = 0
                if "time" in faces_data.dims:
                    time_dim_idx = faces_data.dims.index("time")

                # Directly extract values - should be array of shape [time, num_faces]
                data_array = faces_data.values

                # Ensure time is the first dimension
                if time_dim_idx != 0:
                    dim_order = list(faces_data.dims)
                    dim_order.remove("time")
                    dim_order.insert(0, "time")
                    faces_data = faces_data.transpose(*dim_order)
                    data_array = faces_data.values

                # Store the data array
                batch_data[var_name] = data_array

        # Now create DataFrames using the pre-fetched data
        face_dataframes = {}
        for i, face_idx in enumerate(face_indices):
            data_dict = {}

            # Add center coordinates - repeat for each time step
            data_dict["lat_center"] = [batch_data["lat_center"][i]] * time_dim_len
            data_dict["lon_center"] = [batch_data["lon_center"][i]] * time_dim_len

            # Get node vertex data
            nv_data = batch_data["nv"][i]

            # Get node indices for the three corners of this face
            node_indices = [int(idx - 1) if idx > 0 else int(idx) for idx in nv_data]

            # Extract lat/lon for each corner node
            if "lat_node" in dataset.variables and "lon_node" in dataset.variables:
                for j, node_idx in enumerate(node_indices):
                    corner_num = j + 1  # 1-based for clarity

                    # Add corner lat/lon values using pre-fetched data
                    lat_node_val = float(batch_data["lat_node"][node_idx])
                    lon_node_val = float(batch_data["lon_node"][node_idx])

                    # Add to data dict with repeated values for each time step
                    data_dict[f"element_corner_{corner_num}_lat"] = np.repeat(
                        lat_node_val, time_dim_len
                    )
                    data_dict[f"element_corner_{corner_num}_lon"] = np.repeat(
                        lon_node_val, time_dim_len
                    )

            # Add variable data - ensure proper time dimension
            for var_name in vars_to_include:
                if var_name in vars_to_skip or var_name not in batch_data:
                    continue

                # Check variable dimensions
                if (
                    "sigma_layer" in dataset[var_name].dims
                    and "face" in dataset[var_name].dims
                ):
                    # Handle 3D variables (time, sigma_layer, face)
                    for layer_idx in range(len(dataset.sigma_layer)):
                        col_name = f"{var_name}_layer_{layer_idx}"

                        # Get the time series for this face
                        # This accesses the correct column from our batch data array
                        var_data = batch_data[var_name][layer_idx][:, i]

                        # Verify data length matches time dimension
                        if len(var_data) != time_dim_len:
                            print(
                                f"Warning: {col_name} for face {face_idx} has time dimension {len(var_data)} != {time_dim_len}"
                            )
                            # Skip this variable
                            continue

                        data_dict[col_name] = var_data

                elif (
                    "face" in dataset[var_name].dims
                    and "time" in dataset[var_name].dims
                ):
                    # Handle 2D variables (time, face)
                    # Get the time series for this face
                    var_data = batch_data[var_name][:, i]

                    # Verify data length matches time dimension
                    if len(var_data) != time_dim_len:
                        print(
                            f"Warning: {var_name} for face {face_idx} has time dimension {len(var_data)} != {time_dim_len}"
                        )
                        # Skip this variable
                        continue

                    data_dict[var_name] = var_data

            # Create DataFrame with time as index
            df = pd.DataFrame(data_dict, index=time_values)
            df.index.name = "time"
            face_dataframes[face_idx] = df

        return face_dataframes

    def _prepare_netcdf_compatible_metadata(self, attributes: Dict) -> Dict:
        """
        Process and prepare metadata to be compatible with NetCDF/xarray structure.

        Parameters
        ----------
        attributes : Dict
            Dictionary containing metadata to be included

        Returns
        -------
        Dict
            Dictionary of metadata keys and values converted to bytes
        """

        # Custom JSON encoder to handle NumPy types and other special types
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

        # Add variable attributes
        # For each variable in the DataFrame, store its attributes
        if "variable_attributes" in attributes:
            for var_name, var_attrs in attributes["variable_attributes"].items():
                for attr_name, attr_value in var_attrs.items():
                    # Create keys in format: var_name:attr_name
                    metadata[f"{var_name}:{attr_name}"] = attr_value

        # Add global attributes
        if "global_attributes" in attributes:
            for attr_name, attr_value in attributes["global_attributes"].items():
                # Prefix global attributes to distinguish them
                metadata[f"global:{attr_name}"] = attr_value

        # If attributes is a flat dictionary (not separated into variable/global)
        # Store as-is, assuming they are global attributes
        if not isinstance(attributes, dict) or (
            "variable_attributes" not in attributes
            and "global_attributes" not in attributes
        ):
            for attr_name, attr_value in attributes.items():
                metadata[f"global:{attr_name}"] = attr_value

        # Add metadata markers for parsing when reading the file back
        metadata["_WPTO_HINDCAST_FORMAT_VERSION"] = "1.0"
        metadata["_WPTO_HINDCAST_METADATA_TYPE"] = "netcdf_compatible"

        # Convert all metadata values to strings and then to bytes (required by pyarrow)
        metadata_bytes = {}
        for k, v in metadata.items():
            # Handle different types of values appropriately
            if (
                isinstance(v, (list, dict, tuple))
                or hasattr(v, "__dict__")
                or isinstance(v, np.ndarray)
            ):
                # For complex types, use JSON serialization with NumPy encoder
                try:
                    metadata_bytes[k] = json.dumps(v, cls=NumpyEncoder).encode("utf-8")
                except TypeError as e:
                    # Fallback if JSON encoding fails
                    metadata_bytes[k] = str(v).encode("utf-8")
                    print(
                        f"Warning: Could not JSON encode {k}, using string representation instead: {e}"
                    )
            else:
                # For simple types, use string representation
                metadata_bytes[k] = str(v).encode("utf-8")

        return metadata_bytes

    def _get_face_index_string(self, filename):
        parts = str(filename).split(".")

        for part in parts:
            if "face=" in part:
                return part.replace("face=", "")

        return None

    def _save_parquet_with_metadata(
        self, df: pd.DataFrame, attributes: Dict, partition_path: str, filename: str
    ) -> None:
        """
        Save DataFrame to Parquet with metadata.
        If files with the same face index already exist, concatenate them with the new data.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to save
        attributes : Dict
            Dictionary containing metadata to be included
        partition_path : str
            Path for partitioning
        filename : str
            Output filename
        """
        # Create full directory path for the partition
        full_dir = Path(self.output_dir, partition_path)
        full_dir.mkdir(exist_ok=True, parents=True)

        face_idx = self._get_face_index_string(filename)

        output_filename = filename
        # If we found a face index, check for existing files with the same index
        if face_idx is not None:
            existing_files = sorted(list(full_dir.glob("*.parquet")))
            matching_files = []

            for file in existing_files:
                if f"face={face_idx}" in file.name:
                    matching_files.append(file)

            # If matching files found, concatenate them with the new data
            if len(matching_files) > 0:
                output_filename = matching_files[0].name
                # Use the new filename as the final filename
                # Read and concatenate all existing files
                dfs = [df]  # Start with the new data

                for file in matching_files:
                    existing_df = pd.read_parquet(file)
                    dfs.append(existing_df)
                    # Delete the file after reading
                    file.unlink()

                # Concatenate all dataframes, remove duplicates, and sort
                df = pd.concat(dfs)
                # df = df[~df.index.duplicated(keep="last")]
                df = df.sort_index()

        # Full path to parquet file
        full_path = Path(full_dir, output_filename)

        # Convert DataFrame to pyarrow Table
        table = pa.Table.from_pandas(df)

        # Process metadata using separate method
        metadata_bytes = self._prepare_netcdf_compatible_metadata(attributes)

        # Update table metadata
        table = table.replace_schema_metadata(
            {**table.schema.metadata, **metadata_bytes}
        )

        # Write to parquet
        pq.write_table(table, full_path)

    def convert_dataset(
        self,
        dataset: xr.Dataset,
        vars_to_include: Optional[List[str]] = None,
        max_faces: Optional[int] = None,
    ) -> Dict:
        """
        Convert an xarray Dataset to partitioned Parquet files.

        Parameters
        ----------
        dataset : xr.Dataset
            The input dataset
        vars_to_include : List[str], optional
            List of variable names to include. If None, includes all variables.
        max_faces : int, optional
            Maximum number of faces to process (for testing)

        Returns
        -------
        Dict
            Statistics about the conversion process
        """
        # Extract dataset attributes
        print("Extracting attributes...")
        attributes = self._extract_attributes(dataset)

        # Determine variables to include
        if vars_to_include is None:
            vars_to_include = list(dataset.variables.keys())

        # Get coordinates
        lat_center = dataset.lat_center.values
        lon_center = dataset.lon_center.values

        # Determine the number of faces to process
        num_faces = len(lat_center)
        if max_faces is not None:
            num_faces = min(num_faces, max_faces)

        # Statistics
        stats = {
            "total_faces": num_faces,
            "partitions_created": set(),
            "files_created": 0,
        }

        print("Processing dataframe by face...")

        # Process each face
        for face_idx in range(num_faces):
            print(f"Processing face {face_idx} of {num_faces}...")
            lat = float(lat_center[face_idx])
            lon = float(lon_center[face_idx])

            # Generate partition path
            print("Generating partition path...")
            partition_path = self._get_partition_path(lat, lon)
            print(f"Partition path is: {partition_path}...")

            stats["partitions_created"].add(partition_path)

            # Create time series DataFrame for this face
            print("Creating time series df...")
            df = self._create_time_series_df(dataset, face_idx, vars_to_include)

            # Filename includes face index for uniqueness
            filename = f"face_{face_idx}.parquet"
            if self.config is not None:
                filename = self._get_partition_file_name(face_idx, lat, lon, df)

            # Save to parquet with metadata
            print("Saving parquet file...")
            self._save_parquet_with_metadata(df, attributes, partition_path, filename)
            stats["files_created"] += 1

            print(f"Processed {face_idx}/{num_faces} faces")

        # Convert set to list for easier serialization
        stats["partitions_created"] = list(stats["partitions_created"])

        return stats

    def convert_dataset_batched(
        self,
        dataset: xr.Dataset,
        vars_to_include: Optional[List[str]] = None,
        max_faces: Optional[int] = None,
        batch_size: int = 1000,
    ) -> Dict:
        """
        Convert an xarray Dataset to partitioned Parquet files using batch processing.

        Parameters
        ----------
        dataset : xr.Dataset
            The input dataset
        vars_to_include : List[str], optional
            List of variable names to include. If None, includes all variables.
        max_faces : int, optional
            Maximum number of faces to process (for testing)
        batch_size : int, optional
            Number of faces to process in each batch

        Returns
        -------
        Dict
            Statistics about the conversion process
        """
        # Extract dataset attributes
        print("Extracting attributes...")
        attributes = self._extract_attributes(dataset)

        # Determine variables to include
        if vars_to_include is None:
            vars_to_include = list(dataset.variables.keys())

        # Get coordinates
        lat_center = dataset.lat_center.values
        lon_center = dataset.lon_center.values

        # Determine the number of faces to process
        num_faces = len(lat_center)
        if max_faces is not None:
            num_faces = min(num_faces, max_faces)

        # Statistics
        stats = {
            "total_faces": num_faces,
            "partitions_created": set(),
            "files_created": 0,
        }

        print(f"Processing {num_faces} faces in batches of {batch_size}...")

        # Process faces in batches
        for batch_start in range(0, num_faces, batch_size):
            batch_end = min(batch_start + batch_size, num_faces)
            face_indices = list(range(batch_start, batch_end))

            print(
                f"Processing batch of faces {batch_start} to {batch_end-1} (batch size: {len(face_indices)})"
            )

            # Create time series DataFrames for all faces in this batch
            print("Creating time series dataframes for batch...")
            face_dfs = self._create_time_series_dfs_batch(
                dataset, face_indices, vars_to_include
            )

            # Process and save each face's DataFrame
            print("Saving each face...")
            for face_idx, df in face_dfs.items():
                lat = float(lat_center[face_idx])
                lon = float(lon_center[face_idx])

                # Generate partition path
                partition_path = self._get_partition_path(lat, lon)
                stats["partitions_created"].add(partition_path)

                # Generate filename
                filename = f"face_{face_idx}.parquet"
                if self.config is not None:
                    filename = self._get_partition_file_name(face_idx, lat, lon, df)

                # Save to parquet with metadata
                self._save_parquet_with_metadata(
                    df, attributes, partition_path, filename
                )
                stats["files_created"] += 1

            print(
                f"Processed batch {batch_start}-{batch_end-1} ({stats['files_created']}/{num_faces} faces)"
            )

        print(f"Completed processing all {stats['files_created']} faces")

        # Convert set to list for easier serialization
        stats["partitions_created"] = list(stats["partitions_created"])

        return stats

    @staticmethod
    def _parallel_save_parquet(args):
        """
        Helper function to save a DataFrame to Parquet in parallel.

        Parameters
        ----------
        args : tuple
            Tuple containing (table, full_path)

        Returns
        -------
        str
            Path where the file was saved
        """
        table, full_path = args
        pq.write_table(table, full_path)
        return str(full_path)

    @staticmethod
    def _save_parquet_batch_parallel(batch_tasks):
        """
        Save multiple Parquet files in parallel.

        Parameters
        ----------
        batch_tasks : list
            List of tuples, each containing (table, full_path)

        Returns
        -------
        list
            List of paths where files were saved
        """
        # Use a ThreadPoolExecutor since the IO operations are the bottleneck
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(
                executor.map(ConvertTidalNcToParquet.parallel_save_parquet, batch_tasks)
            )
        return results

    def _save_parquet_with_metadata_parallel(self, write_tasks):
        """
        Prepare and save multiple DataFrames to Parquet with metadata in parallel.

        Parameters
        ----------
        write_tasks : list
            List of tuples, each containing (df, attributes, partition_path, filename)

        Returns
        -------
        list
            Paths of saved files
        """
        batch_tasks = []

        for df, attributes, partition_path, filename in write_tasks:
            # Create full directory path for the partition
            full_dir = Path(self.output_dir, partition_path)
            full_dir.mkdir(exist_ok=True, parents=True)

            face_idx = self._get_face_index_string(filename)
            output_filename = filename

            # Handle existing files with the same face index (still sequential for data consistency)
            if face_idx is not None:
                existing_files = sorted(list(full_dir.glob("*.parquet")))
                matching_files = []

                for file in existing_files:
                    if f"face={face_idx}" in file.name:
                        matching_files.append(file)

                if len(matching_files) > 0:
                    output_filename = matching_files[0].name
                    dfs = [df]

                    for file in matching_files:
                        existing_df = pd.read_parquet(file)
                        dfs.append(existing_df)
                        file.unlink()

                    df = pd.concat(dfs)
                    df = df.sort_index()

            # Full path to parquet file
            full_path = Path(full_dir, output_filename)

            # Convert DataFrame to pyarrow Table
            table = pa.Table.from_pandas(df)

            # Process metadata using separate method
            metadata_bytes = self._prepare_netcdf_compatible_metadata(attributes)

            # Update table metadata
            table = table.replace_schema_metadata(
                {**table.schema.metadata, **metadata_bytes}
            )

            # Add task to batch
            batch_tasks.append((table, full_path))

        # Execute parallel writes
        return ConvertTidalNcToParquet._save_parquet_batch_parallel(batch_tasks)

    def convert_dataset_parallel(
        self,
        dataset,
        vars_to_include=None,
        max_faces=None,
        batch_size=100000,
        write_batch_size=64,
    ):
        """
        Convert an xarray Dataset to partitioned Parquet files using parallel processing.

        Parameters
        ----------
        dataset : xr.Dataset
            The input dataset
        vars_to_include : list, optional
            List of variable names to include. If None, includes all variables.
        max_faces : int, optional
            Maximum number of faces to process (for testing)
        batch_size : int, optional
            Number of faces to process in each batch
        write_batch_size : int, optional
            Number of files to write in parallel

        Returns
        -------
        dict
            Statistics about the conversion process
        """
        # Extract dataset attributes
        print("Extracting attributes...")
        attributes = self._extract_attributes(dataset)

        # Determine variables to include
        if vars_to_include is None:
            vars_to_include = list(dataset.variables.keys())

        # Get coordinates
        lat_center = dataset.lat_center.values
        lon_center = dataset.lon_center.values

        # Determine the number of faces to process
        num_faces = len(lat_center)
        if max_faces is not None:
            num_faces = min(num_faces, max_faces)

        # Statistics
        stats = {
            "total_faces": num_faces,
            "partitions_created": set(),
            "files_created": 0,
        }

        print(f"Processing {num_faces} faces in batches of {batch_size}...")

        # Process faces in batches
        for batch_start in range(0, num_faces, batch_size):
            batch_end = min(batch_start + batch_size, num_faces)
            face_indices = list(range(batch_start, batch_end))

            print(
                f"Processing batch of faces {batch_start} to {batch_end-1} (batch size: {len(face_indices)})"
            )

            # Create time series DataFrames for all faces in this batch
            print("Creating time series dataframes for batch...")
            face_dfs = self._create_time_series_dfs_batch(
                dataset, face_indices, vars_to_include
            )

            # Prepare write tasks
            write_tasks = []
            for face_idx, df in face_dfs.items():
                lat = float(lat_center[face_idx])
                lon = float(lon_center[face_idx])

                # Generate partition path
                partition_path = self._get_partition_path(lat, lon)
                stats["partitions_created"].add(partition_path)

                # Generate filename
                filename = f"face_{face_idx}.parquet"
                if self.config is not None:
                    filename = self._get_partition_file_name(face_idx, lat, lon, df)

                # Collect write tasks
                write_tasks.append((df, attributes, partition_path, filename))

                # When we've collected enough tasks or reached the end, process them in parallel
                if len(write_tasks) >= write_batch_size or face_idx == face_indices[-1]:
                    print(f"Writing batch of {len(write_tasks)} files in parallel...")
                    saved_paths = self._save_parquet_with_metadata_parallel(write_tasks)
                    stats["files_created"] += len(saved_paths)
                    write_tasks = []  # Reset for next batch

            print(
                f"Processed batch {batch_start}-{batch_end-1} ({stats['files_created']}/{num_faces} faces)"
            )

        print(f"Completed processing all {stats['files_created']} faces")

        # Convert set to list for easier serialization
        stats["partitions_created"] = list(stats["partitions_created"])

        return stats


def partition_vap_into_parquet_dataset(config, location_key):
    location = config["location_specification"][location_key]
    input_path = file_manager.get_vap_output_dir(config, location)
    output_path = file_manager.get_vap_partition_output_dir(config, location)

    vap_nc_files = sorted(list(input_path.rglob("*.nc")))

    for nc_file in vap_nc_files:
        converter = ConvertTidalNcToParquet(output_path, config, location)
        ds = xr.open_dataset(nc_file)
        # Access batch_size faces at once
        # This should be set to optimize memory usage and speed
        # A value that is too big will overflow memory, and a value that is too small will take too long
        converter.convert_dataset_parallel(ds, batch_size=100000, write_batch=64)


if __name__ == "__main__":
    converter = ConvertTidalNcToParquet("/scratch/asimms/Tidal/test_parquet")
    print("Reading dataset...")
    ds = xr.open_dataset(
        "/scratch/asimms/Tidal/AK_cook_inlet/b1_vap/001.AK_cook_inlet.tidal_hindcast_fvcom-1h.b1.20050101.000000.nc"
    )

    print("Starting parquet partition creation...")
    converter.convert_dataset(ds, max_faces=10)
