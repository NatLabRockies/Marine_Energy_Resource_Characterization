from pathlib import Path

from typing import List, Dict, Union, Optional

import numpy as np
import xarray as xr
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


class ConvertTidalNcToParquet:
    """
    Converts an xarray Dataset with FVCOM structure to partitioned Parquet files.
    Each face in the dataset will be converted to a time-indexed Parquet file
    stored in a partition based on its lat/lon coordinates.
    """

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

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

        vars_to_include.remove("lat_node")
        vars_to_include.remove("lon_node")

        # Extract data for each variable
        for var_name in vars_to_include:
            print(f"Extracting data for {var_name}")
            var = dataset[var_name]

            # Check variable dimensions
            if "sigma_layer" in var.dims and "face" in var.dims:
                # Handle 3D variables (time, sigma_layer, face)
                for layer_idx in range(len(dataset.sigma_layer)):
                    # Create column name with layer information
                    col_name = f"{var_name}_layer_{layer_idx}"
                    # Extract data for specific face and layer across all times
                    data_dict[col_name] = var.isel(
                        face=face_idx, sigma_layer=layer_idx
                    ).values

            elif "face" in var.dims and "time" in var.dims:
                # Handle 2D variables (time, face)
                data_dict[var_name] = var.isel(face=face_idx).values

            elif "face" in var.dims and "time" not in var.dims:
                # Handle static face variables (e.g., lat_center, lon_center)
                # Repeat the value for each time step
                value = var.isel(face=face_idx).values
                data_dict[var_name] = np.repeat(value, len(time_values))

        # Create DataFrame with time as index
        df = pd.DataFrame(data_dict, index=time_values)
        df.index.name = "time"

        return df

    def _save_parquet_with_metadata(
        self, df: pd.DataFrame, attributes: Dict, partition_path: str, filename: str
    ) -> None:
        """
        Save DataFrame to Parquet with metadata.

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

        # Full path to parquet file
        full_path = Path(full_dir, filename)

        # Convert DataFrame to pyarrow Table
        table = pa.Table.from_pandas(df)

        # Add metadata to the table
        metadata = {"xarray_attributes": str(attributes)}

        # Convert metadata to bytes
        metadata_bytes = {k: str(v).encode("utf-8") for k, v in metadata.items()}

        # Update table metadata
        table = table.replace_schema_metadata(
            {**table.schema.metadata, **metadata_bytes}
        )

        # Write to parquet
        pq.write_table(
            table,
            full_path,
            # compression="snappy",
            # use_deprecated_int96_timestamps=True,  # for better compatibility
        )

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

            # Save to parquet with metadata
            print("Saving parquet file...")
            self._save_parquet_with_metadata(df, attributes, partition_path, filename)
            stats["files_created"] += 1

            print(f"Processed {face_idx}/{num_faces} faces")

        # Convert set to list for easier serialization
        stats["partitions_created"] = list(stats["partitions_created"])

        return stats


if __name__ == "__main__":
    converter = ConvertTidalNcToParquet("/scratch/asimms/Tidal/test_parquet")
    print("Reading dataset...")
    ds = xr.open_dataset(
        "/scratch/asimms/Tidal/AK_cook_inlet/b1_vap/001.AK_cook_inlet.tidal_hindcast_fvcom-1h.b1.20050101.000000.nc"
    )

    print("Starting parquet partition creation...")
    converter.convert_dataset(ds, max_faces=5)
