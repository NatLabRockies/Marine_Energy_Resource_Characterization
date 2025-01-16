from pathlib import Path

import pandas as pd
import xarray as xr
import numpy as np

from . import (
    attrs_manager,
    coord_manager,
    file_name_convention_manager,
    file_manager,
    time_manager,
)


class FVCOMStandardizer:
    """
    A class to standardize FVCOM datasets by applying UGRID conventions.
    First renames dimensions and coordinates to match UGRID spec, then adds appropriate attributes.
    """

    def __init__(self):
        """Initialize the standardizer."""
        self.dim_mapping = {
            "nele": "face",  # FVCOM elements are 2D faces
            "node": "node",  # Already matches UGRID spec
            "three": "face_node_index",  # Number of nodes per face
            "time": "time",  # Keep time dimension as is
            "siglay": "sigma_layer",  # Vertical layers
            "siglev": "sigma_level",  # Vertical levels
        }

        self.coord_mapping = {
            "lat": "lat_node",  # Node (corner) latitude
            "lon": "lon_node",  # Node (corner) longitude
            "latc": "lat_center",  # Face center latitude
            "lonc": "lon_center",  # Face center longitude
            "siglay": "sigma_layer",  # Sigma layer depths
            "siglev": "sigma_level",  # Sigma level depths
        }

    def _rename_dimensions(self, ds):
        """
        Rename dimensions to follow UGRID conventions.

        Parameters:
        -----------
        ds : xarray.Dataset
            Dataset with original FVCOM dimensions

        Returns:
        --------
        xarray.Dataset
            Dataset with standardized dimension names
        """
        rename_dict = {
            old: new for old, new in self.dim_mapping.items() if old in ds.dims
        }
        return ds.rename(rename_dict)

    def _rename_coordinates(self, ds):
        """
        Rename coordinates to follow UGRID conventions.

        Parameters:
        -----------
        ds : xarray.Dataset
            Dataset with original FVCOM coordinates

        Returns:
        --------
        xarray.Dataset
            Dataset with standardized coordinate names
        """
        rename_dict = {
            old: new for old, new in self.coord_mapping.items() if old in ds.variables
        }
        return ds.rename(rename_dict)

    def _add_mesh_topology_attrs(self, ds):
        """
        Add mesh topology attributes following UGRID conventions for a 3D unstructured mesh.
        Uses standardized dimension and coordinate names.
        """
        ds["mesh"] = xr.DataArray(
            attrs={
                "cf_role": "mesh_topology",
                "long_name": "3D unstructured mesh topology",
                "topology_dimension": 3,
                "node_coordinates": "lon_corners lat_corners",
                "face_coordinates": "lon_centers lat_centers",
                "face_node_connectivity": "nv",
                "vertical_coordinates": "sigma_layer sigma_level",
                "coordinate_system": "sigma",
                "face_dimension": "face",
                "node_dimension": "node",
                "vertical_dimension": ["layer", "level"],
            }
        )
        return ds

    def _add_coordinate_attrs(self, ds):
        """Add coordinate attributes following UGRID conventions."""
        # Node coordinates
        if "lat_node" in ds:
            ds.lat_node.attrs.update(
                {
                    "standard_name": "latitude",
                    "long_name": "Nodal Latitude",
                    "units": "degrees_north",
                    "mesh": "mesh",
                }
            )

        if "lon_node" in ds:
            ds.lon_node.attrs.update(
                {
                    "standard_name": "longitude",
                    "long_name": "Nodal Longitude",
                    "units": "degrees_east",
                    "mesh": "mesh",
                }
            )

        # Face coordinates
        if "lat_center" in ds:
            ds.lat_center.attrs.update(
                {
                    "standard_name": "latitude",
                    "long_name": "Face Center Latitude",
                    "units": "degrees_north",
                    "mesh": "mesh",
                }
            )

        if "lon_center" in ds:
            ds.lon_center.attrs.update(
                {
                    "standard_name": "longitude",
                    "long_name": "Face Center Longitude",
                    "units": "degrees_east",
                    "mesh": "mesh",
                }
            )

        return ds

    def _add_vertical_coordinate_attrs(self, ds):
        """Add vertical coordinate attributes following UGRID conventions."""
        if "sigma_layer" in ds:
            ds.sigma_layer.attrs.update(
                {
                    "standard_name": "ocean_sigma/general_coordinate",
                    "long_name": "Sigma Layer",
                    "positive": "up",
                    "valid_min": -1.0,
                    "valid_max": 0.0,
                    "formula_terms": "sigma: sigma_layer eta: zeta depth: h",
                    "mesh": "mesh",
                }
            )

        if "sigma_level" in ds:
            ds.sigma_level.attrs.update(
                {
                    "standard_name": "ocean_sigma/general_coordinate",
                    "long_name": "Sigma Level",
                    "positive": "up",
                    "valid_min": -1.0,
                    "valid_max": 0.0,
                    "formula_terms": "sigma: sigma_level eta: zeta depth: h",
                    "mesh": "mesh",
                }
            )

        return ds

    def add_face_node_connectivity(self, ds):
        """
        Add face node connectivity following UGRID conventions.
        Dynamically determines start_index based on the minimum node index in the dataset.
        """
        # Assuming the original FVCOM connectivity array is called 'nv'
        if "nv" not in ds:
            raise ValueError("Dataset missing required connectivity array 'nv'")

        # Determine start_index dynamically
        start_index = int(ds.nv.min().values)  # Convert from numpy type to Python int

        # Create face_node coordinate with proper attributes
        ds["face_node"] = ds["nv"].rename({"three": "nodes_per_face"})
        ds["face_node"].attrs.update(
            {
                "long_name": "Nodes defining each face",
                "standard_name": "face_node_connectivity",
                "cf_role": "face_node_connectivity",
                "start_index": start_index,  # Dynamically determined
                "mesh": "mesh",
            }
        )

        # Update the mesh topology attributes to reference this connectivity
        if "mesh" in ds:
            ds.mesh.attrs["face_node_connectivity"] = "face_node"

        return ds

    def verify_required_variables(self, ds, required_vars):
        """
        Verify that dataset contains all required variables with correct specifications.

        Checks for existence, data type, dimensions, coordinates, and attributes of all
        required variables in the dataset. Variables can be data variables, coordinates,
        or other variables accessible via ds[key].

        Parameters
        ----------
        ds : xarray.Dataset
            Dataset to verify
        required_vars : dict
            Dictionary specifying required variables and their properties.
            Each variable specification should contain:
                - dtype: str, required data type
                - dimensions: list, required dimension names
                - coordinates: list, required coordinate names
                - attributes: dict, required attributes
                - coverage_content_type: str, optional content type

        Raises
        ------
        KeyError
            If a required variable is missing from the dataset
        ValueError
            If any variable's properties don't match specifications:
            - Incorrect data type
            - Missing or extra dimensions
            - Wrong dimension order
            - Missing or extra coordinates
            - Missing required attributes
        """
        for var_name, var_spec in required_vars.items():
            # Verify variable exists
            try:
                var = ds[var_name]
            except KeyError as _:
                raise KeyError(
                    f"Required variable '{var_name}' missing from dataset. "
                    f"Available variables: {list(ds.variables.keys())}"
                )

            # Verify data type
            expected_dtype = var_spec["dtype"]
            actual_dtype = str(var.dtype)

            # Handle datetime comparison specially
            if expected_dtype.startswith("datetime64"):
                if not actual_dtype.startswith("datetime64"):
                    raise ValueError(
                        f"Variable '{var_name}' has incorrect dtype.\n"
                        f"Expected datetime64 type, got: {actual_dtype}"
                    )
            elif actual_dtype != expected_dtype:
                raise ValueError(
                    f"Variable '{var_name}' has incorrect dtype.\n"
                    f"Expected: {expected_dtype}\n"
                    f"Got: {actual_dtype}"
                )

            # Verify dimensions
            expected_dims = var_spec["dimensions"]
            actual_dims = list(var.dims)

            # Check for missing or extra dimensions
            missing_dims = set(expected_dims) - set(actual_dims)
            extra_dims = set(actual_dims) - set(expected_dims)

            if missing_dims or extra_dims:
                error_msg = f"Variable '{var_name}' has incorrect dimensions.\n"
                if missing_dims:
                    error_msg += f"Missing dimensions: {missing_dims}\n"
                if extra_dims:
                    error_msg += f"Extra dimensions: {extra_dims}\n"
                raise ValueError(error_msg)

            # Check dimension order
            if actual_dims != expected_dims:
                raise ValueError(
                    f"Variable '{var_name}' has incorrect dimension order.\n"
                    f"Expected: {expected_dims}\n"
                    f"Got: {actual_dims}"
                )

            # Verify coordinates
            expected_coords = var_spec["coordinates"]
            actual_coords = list(var.coords)

            # Check for missing or extra coordinates
            missing_coords = set(expected_coords) - set(actual_coords)
            if missing_coords:
                raise ValueError(
                    f"Variable '{var_name}' is missing coordinates: {missing_coords}\n"
                    f"Available coordinates: {actual_coords}"
                )

            # Update attributes, preserving existing ones not in specification
            existing_attrs = dict(var.attrs)  # Make a copy of existing attributes
            required_attrs = var_spec.get("attributes", {})

            # Check for required attributes
            for attr_name, attr_value in required_attrs.items():
                if attr_name not in existing_attrs:
                    existing_attrs[attr_name] = attr_value
                elif existing_attrs[attr_name] != attr_value:
                    raise ValueError(
                        f"Variable '{var_name}' has incorrect value for attribute '{attr_name}'.\n"
                        f"Expected: {attr_value}\n"
                        f"Got: {existing_attrs[attr_name]}"
                    )

            # Add coverage_content_type if specified
            if "coverage_content_type" in var_spec:
                existing_attrs["coverage_content_type"] = var_spec[
                    "coverage_content_type"
                ]

            ds[var_name].attrs = existing_attrs

        return ds

    def clean_variables(self, ds, required_vars):
        """
        Remove non-required variables from dataset, preserving all coordinates and dimensions.

        Parameters
        ----------
        ds : xarray.Dataset
            Dataset to clean
        required_vars : dict
            Dictionary specifying required variables

        Returns
        -------
        xarray.Dataset
            Dataset with only required variables, coordinates, and dimensions
        """
        vars_to_drop = [
            var
            for var in ds.variables
            if var not in required_vars and var not in ds.coords and var not in ds.dims
        ]

        return ds.drop_vars(vars_to_drop)

    def standardize_variables(self, ds, required_vars):
        """
        Update variable attributes and dtypes according to specification.

        Parameters
        ----------
        ds : xarray.Dataset
            Dataset to standardize
        required_vars : dict
            Dictionary specifying required variables and their properties

        Returns
        -------
        xarray.Dataset
            Dataset with standardized attributes and dtypes
        """
        for var_name, var_spec in required_vars.items():
            # Update attributes
            ds[var_name].attrs.clear()
            ds[var_name].attrs.update(var_spec.get("attributes", {}))

            # Set dtype if specified
            if "dtype" in var_spec:
                ds[var_name] = ds[var_name].astype(var_spec["dtype"])

        return ds

    def standardize_coordinate_values(self, ds, location):
        """
        Standardize coordinate values by converting from UTM if necessary and preparing face node coordinates.

        Parameters
        ----------
        ds : xarray.Dataset
            Dataset containing FVCOM coordinate variables (lat, lon, latc, lonc)
        location : dict
            Dictionary containing coordinate system information with structure:
            {
                'coordinates': {
                    'system': str,  # 'utm' or other
                    'zone': int,    # UTM zone number if applicable
                }
            }

        Returns
        -------
        tuple
            - xarray.Dataset: Dataset with standardized coordinate values
            - dict: Face node coordinates with keys 'lat_face_nodes' and 'lon_face_nodes'
        """

        utm_zone = None
        if location["coordinates"]["system"] == "utm":
            utm_zone = location["coordinates"]["zone"]

        coords = coord_manager.standardize_fvcom_coords(ds, utm_zone)

        # Update existing coordinates
        ds.latc.values = coords["lat_centers"]
        ds.lonc.values = coords["lon_centers"]
        ds.lat.values = coords["lat_nodes"]
        ds.lon.values = coords["lon_nodes"]

        face_nodes = {
            "lat_face_nodes": coords["lat_face_nodes"],
            "lon_face_nodes": coords["lon_face_nodes"],
        }

        return ds, face_nodes

    def add_face_nodes_dimension(self, ds, face_nodes):
        """
        Add face node coordinates to the dataset with UGRID-compliant dimensions and attributes.
        Associates nodes with faces through a connectivity array following UGRID conventions.

        Parameters
        ----------
        ds : xarray.Dataset
            Dataset to add face node coordinates to
        face_nodes : dict
            Dictionary containing face node coordinates with keys:
            - 'lat_face_nodes': numpy.ndarray of face node latitudes
            - 'lon_face_nodes': numpy.ndarray of face node longitudes

        Returns
        -------
        xarray.Dataset
            Dataset with added face node coordinates following UGRID conventions
        """
        # Add the face_nodes dimension
        ds["face_node_index"] = xr.DataArray(
            [0, 1, 2],  # The three nodes of each triangular face
            dims=["face_node"],
            attrs={
                "long_name": "Index of node within face",
                "cf_role": "face_node_connectivity",
                "start_index": 0,
            },
        )

        # Create the face node coordinates
        ds["lat_face_node"] = xr.DataArray(
            face_nodes["lat_face_nodes"],
            dims=["face", "face_node"],
            coords={"face_node_index": ds.face_node_index},
            attrs={
                "standard_name": "latitude",
                "long_name": "Face Node Latitude",
                "units": "degrees_north",
                "mesh": "mesh",
                "location": "face_node",
                "coverage_content_type": "coordinate",
                "valid_min": "-90",
                "valid_max": "90",
            },
        )

        ds["lon_face_node"] = xr.DataArray(
            face_nodes["lon_face_nodes"],
            dims=["face", "face_node"],
            coords={"face_node_index": ds.face_node_index},
            attrs={
                "standard_name": "longitude",
                "long_name": "Face Node Longitude",
                "units": "degrees_east",
                "mesh": "mesh",
                "location": "face_node",
                "coverage_content_type": "coordinate",
                "valid_min": "-180",
                "valid_max": "180",
            },
        )

        # Add mesh topology attributes to make the relationship clear
        ds["mesh"].attrs.update(
            {
                "face_node_connectivity": "face_node_index",
                "face_coordinates": "lon_center lat_center",
                "face_node_coordinates": "lon_face_node lat_face_node",
            }
        )

        return ds

    def standardize_time(self, ds, corrected_timestamps):
        print("Correcting `time`")
        ds["time"] = corrected_timestamps
        ds["time"].attrs = {
            "standard_name": "time",
            "long_name": "Time",
            "axis": "T",
            "calendar": "proleptic_gregorian",
            # "units": "nanoseconds since 1970-01-01T00:00:00Z",  # Explicit nanosecond units
            "precision": "nanosecond",
        }

        return ds

    def standardize(self, ds_path, config, location, corrected_timestamps):
        """
        Standardize an FVCOM dataset by verifying required variables and adding UGRID conventions.

        Parameters:
        -----------
        ds : xarray.Dataset
            The FVCOM dataset to standardize
        required_vars : dict
            Dictionary specifying required variables and their properties

        Returns:
        --------
        xarray.Dataset
            Dataset with standardized names and UGRID attributes
        """
        ds = xr.open_dataset(ds_path, decode_times=False)

        ds = self.standardize_time(ds, corrected_timestamps)
        ds, face_nodes = self.standardize_coordinate_values(ds, location)

        required_vars = config["standardized_variable_specification"]

        # Then rename dimensions and coordinates
        ds = self._rename_dimensions(ds)
        ds = self._rename_coordinates(ds)

        # Then add UGRID convention attributes
        ds = self._add_mesh_topology_attrs(ds)
        ds = self._add_coordinate_attrs(ds)
        ds = self._add_vertical_coordinate_attrs(ds)

        ds = self.add_face_nodes_dimension(ds, face_nodes)

        # Verify all required variables exist
        self.verify_required_variables(ds, required_vars)

        # Then clean and standardize variables
        ds = self.clean_variables(ds, required_vars)

        return ds


class DatasetStandardizer:
    def __init__(self, config, location_key):
        self.config = config
        self.location_key = location_key
        self.location = config["location_specification"][location_key]
        self.variable_mapping = self._build_variable_mapping()

    def _build_variable_mapping(self):
        return self.config["model_specification"]["required_original_variables"]

    def _create_base_coords(self, coords_dict):
        ds = xr.Dataset(
            coords={
                "time": (
                    "time",
                    coords_dict["time"],
                    {
                        "standard_name": "time",
                        "axis": "T",
                    },
                ),
                "fvcom_node": (
                    "fvcom_node",
                    np.arange(coords_dict["lat_corners"].shape[1]),
                    {
                        "standard_name": "mesh_node",
                        "long_name": "Mesh node",
                        "cf_role": "mesh_node",
                    },
                ),
                "fvcom_face": (
                    "fvcom_face",
                    np.arange(len(coords_dict["lat_centers"])),
                    {
                        "standard_name": "mesh_face",
                        "long_name": "Mesh face",
                        "cf_role": "mesh_face",
                    },
                ),
                "sigma": (
                    "sigma",
                    coords_dict["sigma_levels"],
                    {
                        "standard_name": "ocean_sigma_coordinate",
                        "long_name": "Sigma layer",
                        "positive": "up",
                        "formula_terms": "sigma: sigma eta: zeta depth: h_center",
                    },
                ),
            }
        )
        return ds

    def _create_cf_compliant_coords(self, ds, coords_dict):
        # Node coordinates
        ds["fvcom_node_x"] = xr.DataArray(
            data=coords_dict["lon_corners"],
            dims=["fvcom_node"],
            attrs={
                "standard_name": "longitude",
                "long_name": "Longitude of mesh nodes",
                "units": "degrees_east",
                "mesh": "fvcom",
            },
        )

        ds["fvcom_node_y"] = xr.DataArray(
            data=coords_dict["lat_corners"],
            dims=["fvcom_node"],
            attrs={
                "standard_name": "latitude",
                "long_name": "Latitude of mesh nodes",
                "units": "degrees_north",
                "mesh": "fvcom",
            },
        )

        # Face (element center) coordinates
        ds["fvcom_face_x"] = xr.DataArray(
            data=coords_dict["lon_centers"],
            dims=["fvcom_face"],
            attrs={
                "standard_name": "longitude",
                "long_name": "Longitude of mesh face centers",
                "units": "degrees_east",
                "mesh": "fvcom",
            },
        )

        ds["fvcom_face_y"] = xr.DataArray(
            data=coords_dict["lat_centers"],
            dims=["fvcom_face"],
            attrs={
                "standard_name": "latitude",
                "long_name": "Latitude of mesh face centers",
                "units": "degrees_north",
                "mesh": "fvcom",
            },
        )

        return ds

    def _add_mesh_topology(self, ds):
        ds["fvcom"] = xr.DataArray(
            attrs={
                "cf_role": "mesh_topology",
                "long_name": "Mesh topology",
                "topology_dimension": 2,
                "node_coordinates": "fvcom_node_x fvcom_node_y",
                "face_coordinates": "fvcom_face_x fvcom_face_y",
                "face_node_connectivity": "fvcom_face_nodes",
                "coordinate_reference_system": "WGS84",
            }
        )
        return ds

    def _setup_vertical_coordinates(self, ds, sigma_levels):
        ds["sigma"] = xr.DataArray(
            data=sigma_levels,
            dims=["sigma"],
            attrs={
                "standard_name": "ocean_sigma_coordinate",
                "long_name": "Sigma layer centers",
                "positive": "up",
                "formula_terms": "sigma: sigma eta: zeta depth: h",
                "mesh": "fvcom",
            },
        )
        return ds

    def _interpolate_zeta_to_cell_centers(self, ds):
        """Interpolate zeta from nodes to cell centers"""
        node_to_cell_map = coord_manager.get_node_to_cell_mapping(ds)
        cell_values = np.zeros((ds.zeta.shape[0], len(node_to_cell_map)))

        for t in range(ds.zeta.shape[0]):
            node_values = ds.zeta[t].values
            cell_values[t] = np.mean(node_values[node_to_cell_map], axis=1)

        return cell_values

    def _add_variables(self, ds, orig_ds):
        for var_name, var_specs in self.variable_mapping.items():
            print(f"Adding {var_name}...")
            if var_name == "u" or var_name == "v":
                coords = {
                    "time": ds.time,
                    "sigma": ds.sigma,
                    "fvcom_face": ds.fvcom_face,
                }
                dims = ["time", "sigma", "fvcom_face"]
                data = orig_ds[var_name].values
                new_var_name = var_name
            elif var_name == "zeta":
                coords = {
                    "time": ds.time,
                    "fvcom_node": ds.fvcom_node,
                }
                dims = ["time", "fvcom_node"]
                # data = self._interpolate_zeta_to_cell_centers(orig_ds)
                data = orig_ds[var_name].values
                new_var_name = "zeta"
            elif var_name == "h_center":
                coords = {
                    "cell": ds.cell,
                }
                dims = ["cell"]
                data = orig_ds[var_name].values
                new_var_name = var_name
            elif var_name == "siglev_center":
                # Skip sigma, it already exists
                continue
                # coords = ["sigma"]
            else:
                raise ValueError(f"Coordinates not defined for var: {var_name}")

            ds[new_var_name] = xr.DataArray(
                data=data,
                coords=coords,
                dims=dims,
                attrs={
                    **orig_ds[var_name].attrs,
                    "coverage_content_type": "modelResult",
                    "grid_mapping": "mesh",
                },
            )

        return ds

    def calculate_siglay_center(self, siglev_center):
        # Take the average of consecutive points
        siglay_center = (siglev_center[:-1] + siglev_center[1:]) / 2
        return siglay_center

    def _extract_sigma_layer(self, ds):
        if "siglay_center" in ds:
            # Get the first column as reference
            reference_column = ds["siglay_center"].values[:, 0]

            # Broadcast and compare all columns against the first column at once
            # This creates a bool array of shape (n_layers, n_elements-1)
            columns_match = np.allclose(
                ds["siglay_center"].values[:, 1:],
                reference_column[:, np.newaxis],
                rtol=1e-10,
                atol=1e-10,
            )

            if not columns_match:
                raise ValueError("Not all columns in siglay_center are identical")

            return reference_column

        return self.calculate_siglay_center(ds["siglev_center"].values[:0])

    def standardize_single_file(self, source_file, time_df):
        print(f"Opening source file: {source_file}...")
        ds = xr.open_dataset(source_file, decode_times=False)
        utm_zone = None
        if self.location["coordinates"]["system"] == "utm":
            utm_zone = self.location["coordinates"]["zone"]

        print("Standardizing coords...")
        coords = coord_manager.standardize_fvcom_coords(ds, utm_zone)

        print("Extracting sigma_layer...")
        sigma_layers = self._extract_sigma_layer(ds)

        print("Creating new ds...")
        new_ds = self._create_base_coords(
            {
                **coords,
                "time": time_df["timestamp"].values,
                "sigma_levels": sigma_layers,
            }
        )
        print("Creating cf_compliant")
        new_ds = self._create_cf_compliant_coords(new_ds, coords)
        print("Setup vert coords...")
        new_ds = self._setup_vertical_coordinates(new_ds, sigma_layers)
        print("Adding mesh topology...")
        new_ds = self._add_mesh_topology(new_ds)
        print("Adding variables ...")
        new_ds = self._add_variables(new_ds, ds.sel(time=time_df["original"].values))
        return new_ds


def standardize_dataset(config, location_key, valid_timestamps_df):
    # standardizer = DatasetStandardizer(config, location_key)
    standardizer = FVCOMStandardizer()

    # Check for existing standardization file
    location = config["location_specification"][location_key]
    tracking_folder = file_manager.get_tracking_output_dir(config, location)
    output_name = config["location_specification"][location_key]["output_name"]
    tracking_path = Path(
        tracking_folder,
        f"{output_name}_standardize_step_tracking.parquet",
    )

    # Check if tracking file exists
    if tracking_path.exists():
        print(f"\tDataset already verified: {output_name}")
        return pd.read_parquet(tracking_path)

    drop_strategy = config["time_specification"][
        "drop_duplicate_timestamps_keep_strategy"
    ]
    time_df = valid_timestamps_df.drop_duplicates(keep=drop_strategy)
    spec_start_date = pd.to_datetime(
        config["location_specification"][location_key]["start_date"], utc=True
    )
    spec_end_date = pd.to_datetime(
        config["location_specification"][location_key]["end_date"], utc=True
    )

    # Filtering timestamps between specified start and end dates (inclusive)
    time_df = time_df[
        (time_df["timestamp"] >= spec_start_date)
        & (time_df["timestamp"] <= spec_end_date)
    ]

    time_manager.does_time_match_specification(
        time_df["timestamp"], location["expected_delta_t_seconds"]
    )
    std_files = []

    count = 1

    # One to one go through the source files and standardize them
    for source_file, this_df in time_df.groupby("source_file"):
        print(f"Processing file {count}: {source_file}")
        print(f"Number of timestamps: {len(this_df)}")
        print(f"Start time: {this_df['timestamp'].iloc[0]}")
        print(f"End time: {this_df['timestamp'].iloc[-1]}")
        print("Performing standardization...")

        output_ds = standardizer.standardize(
            source_file, config, location, this_df["timestamp"].values
        )

        # print("Correcting `time`")
        # output_ds["time"] = this_df["timestamp"].values
        # output_ds["time"].attrs = {
        #     "standard_name": "time",
        #     "long_name": "time",
        #     "axis": "T",
        #     "calendar": "proleptic_gregorian",
        #     "units": "nanoseconds since 1970-01-01T00:00:00Z",  # Explicit nanosecond units
        #     "precision": "nanosecond",
        # }

        print("Finished Standardization!...")
        print("Adding Global Attributes...")
        output_ds = attrs_manager.standardize_dataset_global_attrs(
            output_ds,
            config,
            location,
            "a1",
            [str(f) for f in [source_file]],
            input_ds_is_original_model_output=True,
            coordinate_reference_system_string=coord_manager.OUTPUT_COORDINATE_REFERENCE_SYSTEM.to_string(),
        )

        expected_delta_t_seconds = location["expected_delta_t_seconds"]
        if expected_delta_t_seconds == 3600:
            temporal_string = "1h"
        elif expected_delta_t_seconds == 1800:
            temporal_string = "30m"
        else:
            raise ValueError(
                f"Unexpected expected_delta_t_seconds configuration {expected_delta_t_seconds}"
            )

        data_level_file_name = (
            file_name_convention_manager.generate_filename_for_data_level(
                output_ds,
                output_name,
                config["dataset"]["name"],
                "a1",
                temporal=temporal_string,
            )
        )

        output_path = Path(
            file_manager.get_standardized_output_dir(config, location),
            f"{count:03d}.{data_level_file_name}",
        )
        std_files.extend([output_path] * len(this_df))
        output_ds.to_netcdf(output_path)
        print(f"Saving standardized dataframe to {output_path}...")
        count += 1

    time_df["std_files"] = [str(f) for f in std_files]

    # Save standardization results to parquet
    time_df.to_parquet(tracking_path)

    return time_df
