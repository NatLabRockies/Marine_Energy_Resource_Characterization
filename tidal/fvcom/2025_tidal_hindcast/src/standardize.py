import os
import socket
import platform

from pathlib import Path

import pandas as pd
import xarray as xr
import numpy as np

from . import coord_manager, file_manager, time_manager, version


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
                "cell": (
                    "cell",
                    np.arange(len(coords_dict["lat_centers"])),
                    {"long_name": "Cell Index", "axis": "N", "cf_role": "cell_index"},
                ),
                "vertex": (
                    "vertex",
                    np.arange(coords_dict["lat_corners"].shape[1]),
                    {"long_name": "Vertex Index", "cf_role": "vertex_index"},
                ),
                "sigma": (
                    "sigma",
                    coords_dict["sigma_levels"],
                    {
                        "long_name": "Sigma Layer",
                        "positive": "up",
                        "standard_name": "ocean_sigma_coordinate",
                    },
                ),
            }
        )
        return ds

    def _create_cf_compliant_coords(self, ds, coords_dict):
        ds["latitude"] = xr.DataArray(
            data=coords_dict["lat_centers"],
            dims=["cell"],
            attrs={
                "standard_name": "latitude",
                "long_name": "Cell Center Latitude",
                "units": "degrees_north",
                "axis": "Y",
                "coverage_content_type": "coordinate",
                "grid_mapping": "mesh",
            },
        )

        ds["longitude"] = xr.DataArray(
            data=coords_dict["lon_centers"],
            dims=["cell"],
            attrs={
                "standard_name": "longitude",
                "long_name": "Cell Center Longitude",
                "units": "degrees_east",
                "axis": "X",
                "coverage_content_type": "coordinate",
                "grid_mapping": "mesh",
            },
        )

        ds["latitude_vertices"] = xr.DataArray(
            data=coords_dict["lat_corners"],
            dims=["cell", "vertex"],
            attrs={
                "standard_name": "latitude",
                "long_name": "Cell Vertex Latitudes",
                "units": "degrees_north",
                "coverage_content_type": "coordinate",
                "grid_mapping": "mesh",
            },
        )

        ds["longitude_vertices"] = xr.DataArray(
            data=coords_dict["lon_corners"],
            dims=["cell", "vertex"],
            attrs={
                "standard_name": "longitude",
                "long_name": "Cell Vertex Longitudes",
                "units": "degrees_east",
                "coverage_content_type": "coordinate",
                "grid_mapping": "mesh",
            },
        )
        return ds

    def _setup_vertical_coordinates(self, ds, sigma_levels):
        ds["sigma"] = xr.DataArray(
            data=sigma_levels,
            dims=["sigma"],
            attrs={
                "standard_name": "ocean_sigma_coordinate",
                "long_name": "Sigma Layer Centers",
                "positive": "up",
                "formula_terms": "sigma: sigma eta: zeta depth: h_center",
                "computed_standard_name": "depth",
                "coverage_content_type": "coordinate",
            },
        )
        return ds

    def _add_mesh_topology(self, ds):
        ds["mesh"] = xr.DataArray(
            data=np.ones(len(ds.cell)),
            dims=["cell"],
            attrs={
                "cf_role": "mesh_topology",
                "topology_dimension": 2,
                "node_coordinates": "latitude longitude",
                "face_coordinates": "latitude_vertices longitude_vertices",
                "coordinate_reference_system": "WGS84",
                "long_name": "Mesh Topology",
                "coverage_content_type": "modelResult",
            },
        )
        return ds

    def _add_variables(self, ds, orig_ds):
        for var_name, var_specs in self.variable_mapping.items():
            print(ds.orig_ds)
            if var_name == "u" or var_name == "v":
                coords = ["time", "sigma", "cell"]
            elif var_name == "zeta":
                coords = ["time", "cell"]
            elif var_name == "h_center":
                coords = ["cell"]
            elif var_name == "siglay_center":
                # Skip sigma, it already exists
                continue
                # coords = ["sigma"]
            else:
                raise ValueError(f"Coordinates not defined for var: {var_name}")

            ds[var_name] = xr.DataArray(
                data=orig_ds[var_name].values,
                coords=coords,
                dims=coords,
                attrs={
                    **orig_ds[var_name].attrs,
                    "coverage_content_type": "modelResult",
                    "grid_mapping": "mesh",
                },
            )

        return ds

    def _extract_sigma_layer(self, ds):
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

    def standardize_single_file(self, source_file, time_df):
        ds = xr.open_dataset(source_file, decode_times=False)
        coords = coord_manager.standardize_fvcom_coords(ds)
        sigma_layers = self._extract_sigma_layer(ds)
        new_ds = self._create_base_coords(
            {
                **coords,
                "time": time_df["timestamp"].values,
                "sigma_levels": sigma_layers,
            }
        )
        new_ds = self._create_cf_compliant_coords(new_ds, coords)
        new_ds = self._setup_vertical_coordinates(new_ds, sigma_layers)
        new_ds = self._add_mesh_topology(new_ds)
        new_ds = self._add_variables(new_ds, ds.sel(time=time_df["original"].values))
        return new_ds


class DatasetFinalizer:
    @staticmethod
    def _get_system_metadata() -> dict:
        return {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "processor": platform.processor(),
        }

    @staticmethod
    def _get_conda_info() -> dict:
        try:
            import subprocess
            import json

            env_name = os.getenv("CONDA_DEFAULT_ENV", "unknown")
            conda_version = subprocess.run(
                ["conda", "--version"], capture_output=True, text=True
            ).stdout.strip()

            conda_list = subprocess.run(
                ["conda", "list", "--json"], capture_output=True, text=True
            )
            packages = json.loads(conda_list.stdout)
            package_str = ", ".join(
                f"{pkg['name']}={pkg['version']}" for pkg in packages
            )

            return {
                "conda_environment": env_name,
                "conda_version": conda_version,
                "conda_packages": package_str,
            }
        except Exception as e:
            return {
                "conda_environment": "error_getting_conda_info",
                "conda_version": "error_getting_conda_info",
                "conda_packages": f"error_getting_conda_info: {str(e)}",
            }

    @staticmethod
    def add_global_attributes(
        ds,
        config,
        location,
        source_files,
        version,
    ):
        existing_metadata = {
            f"original_{key}": value for key, value in ds.attrs.items()
        }

        new_metadata = {
            **existing_metadata,
            **config["metadata"],
            **location,
            **config["model_specification"]["model_metadata"],
            "processing_timestamp": pd.Timestamp.now(tz="UTC").isoformat(),
            "processing_user": os.getenv("USER", "unknown"),
            "processing_software_version": version,
            **DatasetFinalizer._get_system_metadata(),
            **DatasetFinalizer._get_conda_info(),
            "source_file_count": len(source_files),
            "source_files": ", ".join(source_files),
            "Conventions": "CF-1.0",
            "institution": "PNNL",
            "references": "FVCOM Manual",
            "source": "FVCOM 4.3.1 model output",
            "history": f'Created {pd.Timestamp.now(tz="UTC").isoformat()}',
            "mesh_type": "unstructured",
            "coordinate_reference_system": "WGS84 (EPSG:4326)",
            "geospatial_vertical_positive": "up",
        }

        ds.attrs = new_metadata
        return ds


def standardize_dataset(config, location_key, valid_timestamps_df):
    standardizer = DatasetStandardizer(config, location_key)

    drop_strategy = config["time_specification"][
        "drop_duplicate_timestamps_keep_strategy"
    ]
    time_df = valid_timestamps_df.drop_duplicates(keep=drop_strategy)

    time_manager.does_time_match_specification(
        time_df["timestamp"], standardizer.location["expected_delta_t_seconds"]
    )

    std_files = []

    # One to one go through the source files and standardize them
    for source_file, this_df in time_df.groupby("source_file"):
        print(f"Processing file: {source_file}")
        print(f"Number of timestamps: {len(this_df)}")
        print(f"Start time: {this_df['timestamp'].iloc[0]}")
        print(f"End time: {this_df['timestamp'].iloc[-1]}")

        output_ds = standardizer.standardize_single_file(source_file, this_df)

        output_ds = DatasetFinalizer.add_global_attributes(
            ds=output_ds,
            config=config,
            location=standardizer.location,
            source_files=[str(f) for f in [source_file]],
            version=version.version,
        )

        output_path = Path(
            file_manager.get_standardized_output_dir(config),
            f"{standardizer.location['output_name']}_{Path(source_file).name}_std.nc",
        )
        std_files.append(output_path)
        output_ds.to_netcdf(output_path)

        exit()

    return output_ds
