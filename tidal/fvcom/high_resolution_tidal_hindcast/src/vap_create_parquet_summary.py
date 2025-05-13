from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from . import file_manager, file_name_convention_manager


def convert_tidal_summary_nc_to_dataframe(ds):
    face_vars = []
    sigma_vars = []
    other_vars = []

    for var_name in ds.data_vars:
        dims = ds[var_name].dims
        if len(dims) == 1 and "face" in dims:
            face_vars.append(var_name)
        elif len(dims) == 2 and "sigma_layer" in dims and "face" in dims:
            sigma_vars.append(var_name)
        else:
            other_vars.append(var_name)

    print(f"Face-only variables: {len(face_vars)}")
    print(f"Sigma layer variables: {len(sigma_vars)}")
    print(f"Other variables: {len(other_vars)}")

    print("Creating dataframe with all variables...")

    # Create a dictionary of all data we want to include
    data_dict = {
        "lat_center": ds.lat_center.values,
        "lon_center": ds.lon_center.values,
    }

    # Add element corner coordinates
    print("Adding element corner coordinates...")
    nv = ds.nv.values.T - 1  # Adjust for 0-based indexing if needed

    # Extract corner coordinates
    for i in range(3):  # Each element has 3 corners
        # Get node indices for this corner
        corner_indices = nv[:, i]

        # Add lat/lon for each corner
        data_dict[f"element_corner_{i+1}_lat"] = ds.lat_node.values[corner_indices]
        data_dict[f"element_corner_{i+1}_lon"] = ds.lon_node.values[corner_indices]

    # Add face-only variables
    if face_vars:
        print(f"Adding {len(face_vars)} face-only variables...")
        for var_name in face_vars:
            data_dict[var_name] = ds[var_name].values

    # Add sigma layer variables with suffixes
    if sigma_vars:
        print(
            f"Adding {len(sigma_vars)} sigma layer variables with sigma level suffixes..."
        )
        n_sigma = len(ds.sigma_layer)

        for var_name in sigma_vars:
            for sigma_idx in range(n_sigma):
                sigma_level = sigma_idx + 1
                column_name = f"{var_name}_sigma_level_{sigma_level}"
                data_dict[column_name] = ds[var_name].values[sigma_idx, :]

    # Create dataframe all at once
    result_df = pd.DataFrame(data_dict)

    print(f"Created dataframe with {result_df.shape[1]} columns")

    return result_df


def convert_nc_summary_to_parquet(config, location_key):
    location = config["location_specification"][location_key]
    input_path = file_manager.get_yearly_summary_vap_output_dir(config, location)
    output_path = file_manager.get_vap_summary_parquet_dir(config, location)

    input_nc_files = sorted(list(input_path.rglob("*.nc")))

    for nc_file in input_nc_files:
        ds = xr.open_dataset(nc_file)
        output_df = convert_tidal_summary_nc_to_dataframe(ds)

        # 001.AK_cook_inlet.tidal_hindcast_fvcom-1_year_average.b2.20050101.000000.nc
        # Get the last 2 parts of the filename
        date_time_parts = nc_file.name.split(".")[-3:-1]

        output_filename = file_name_convention_manager.generate_filename_for_data_level(
            output_df,
            location["output_name"],
            config["dataset"]["name"],
            "b4",
            temporal="year_average",
            ext="parquet",
            static_time=date_time_parts,
        )

        output_df.to_parquet(Path(output_path, output_filename))

        for col in output_df.columns:
            print(f"{col}: {output_df[col].dtype}")

        # # Temp rename puget sound cols for atlas
        # if location_key == "puget_sound":
        #     rename_map = {
        #         "speed_depth_avg": "vap_water_column_mean_sea_water_speed",
        #         "power_density_depth_avg": "vap_water_column_mean_sea_water_power_density",
        #     }
        #     output_df = output_df.rename(rename_map, axis="columns")

        cols_for_atlas = [
            "lat_center",
            "lon_center",
            "element_corner_1_lat",
            "element_corner_1_lon",
            "element_corner_2_lat",
            "element_corner_2_lon",
            "element_corner_3_lat",
            "element_corner_3_lon",
            "vap_water_column_mean_sea_water_speed",
            "vap_water_column_95th_percentile_sea_water_speed",
            "vap_water_column_max_sea_water_speed",
            "vap_water_column_mean_sea_water_power_density",
            "vap_water_column_95th_percentile_sea_water_power_density",
            "vap_water_column_max_sea_water_power_density",
        ]

        atlas_df = output_df[cols_for_atlas]

        output_path = file_manager.get_vap_atlas_summary_parquet_dir(config, location)
        output_filename = file_name_convention_manager.generate_filename_for_data_level(
            output_df,
            location["output_name"],
            config["dataset"]["name"],
            "b5",
            temporal="year_average",
            ext="parquet",
            static_time=date_time_parts,
        )

        atlas_df.to_parquet(Path(output_path, output_filename))
