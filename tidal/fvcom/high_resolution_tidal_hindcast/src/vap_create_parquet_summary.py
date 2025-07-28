import shutil

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

from shapely.geometry import Point, Polygon

from . import file_manager, file_name_convention_manager

POLYGON_COLUMNS = {
    "element_corner_1_lat": "Element Corner 1 Latitude",
    "element_corner_1_lon": "Element Corner 1 Longitude",
    "element_corner_2_lat": "Element Corner 2 Latitude",
    "element_corner_2_lon": "Element Corner 2 Longitude",
    "element_corner_3_lat": "Element Corner 3 Latitude",
    "element_corner_3_lon": "Element Corner 3 Longitude",
}

ATLAS_COLUMNS = {
    "lat_center": "Center Latitude",
    "lon_center": "Center Longitude",
    "vap_water_column_mean_sea_water_speed": "Mean Sea Water Speed [m/s]",
    "vap_water_column_95th_percentile_sea_water_speed": "95th Percentile Sea Water Speed [m/s]",
    "vap_water_column_sea_water_speed_max_to_mean_ratio": "Speed Max to Mean Ratio",
    "vap_water_column_mean_sea_water_power_density": "Mean Sea Water Power Density [W/m^2]",
    "vap_water_column_95th_percentile_sea_water_power_density": "95th Percentile Sea Water Power Density [W/m^2]",
    # "vap_water_column_sea_water_power_density_max_to_mean_ratio": "Power Density Max to Mean Ratio",
    "vap_grid_resolution": "Grid Resolution [m]",
    "vap_sea_floor_depth": "Sea Floor Depth from Mean Surface Elevation [m]",
}

GDF_POLYGON_COLUMNS = [col for col in ATLAS_COLUMNS.keys() if "element_corner" in col]


def compute_grid_resolution(df):
    """Compute FVCOM grid resolution as average edge length"""

    # Convert degrees to radians
    lat1_rad = np.radians(df["element_corner_1_lat"])
    lon1_rad = np.radians(df["element_corner_1_lon"])
    lat2_rad = np.radians(df["element_corner_2_lat"])
    lon2_rad = np.radians(df["element_corner_2_lon"])
    lat3_rad = np.radians(df["element_corner_3_lat"])
    lon3_rad = np.radians(df["element_corner_3_lon"])

    # Earth radius in meters (WGS84)
    R = 6378137.0

    def haversine_vectorized(lat1, lon1, lat2, lon2):
        """Vectorized haversine distance calculation"""
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(a))
        return R * c

    # Calculate three edge lengths
    edge1 = haversine_vectorized(lat1_rad, lon1_rad, lat2_rad, lon2_rad)
    edge2 = haversine_vectorized(lat2_rad, lon2_rad, lat3_rad, lon3_rad)
    edge3 = haversine_vectorized(lat3_rad, lon3_rad, lat1_rad, lon1_rad)

    # Grid resolution as average edge length, units are meters
    df["vap_grid_resolution"] = (edge1 + edge2 + edge3) / 3

    return df


def compute_max_to_mean_ratio(df):
    # Convert to consistent float64 before division
    # TODO: Fix this in the calculation of these values in the first place
    speed_95th = df["vap_water_column_95th_percentile_sea_water_speed"].astype(
        "float64"
    )
    power_95th = df["vap_water_column_95th_percentile_sea_water_power_density"].astype(
        "float64"
    )

    df["vap_water_column_sea_water_speed_max_to_mean_ratio"] = (
        speed_95th / df["vap_water_column_mean_sea_water_speed"]
    )
    df["vap_water_column_sea_water_power_density_max_to_mean_ratio"] = (
        power_95th / df["vap_water_column_mean_sea_water_power_density"]
    )

    return df


def create_geo_dataframe(df, geometry_type="polygon"):
    """
    Create a GeoDataFrame from FVCOM DataFrame

    Args:
        df: DataFrame with FVCOM data (must have lat_center, lon_center, element_corner_*_lat/lon)
        geometry_type: 'polygon' for triangular elements or 'point' for center points

    Returns:
        GeoDataFrame with appropriate geometry
    """

    print(
        f"Creating GeoDataFrame with {geometry_type} geometry for {len(df)} elements..."
    )

    if geometry_type == "polygon":
        # Create triangular polygons from element corners
        print("  Building polygon geometries from element corners...")

        def make_triangle(row):
            coords = [
                (row["element_corner_1_lon"], row["element_corner_1_lat"]),
                (row["element_corner_2_lon"], row["element_corner_2_lat"]),
                (row["element_corner_3_lon"], row["element_corner_3_lat"]),
                (
                    row["element_corner_1_lon"],
                    row["element_corner_1_lat"],
                ),  # Close triangle
            ]
            return Polygon(coords)

        # Create geometry series first to avoid future warning
        geometry_series = df.apply(make_triangle, axis=1)
        gdf = gpd.GeoDataFrame(df.copy(), geometry=geometry_series)

        # Filter columns use to create geometry that are not needed in the GeoDataFrame
        gdf = gdf.drop(
            [
                "element_corner_1_lon",
                "element_corner_1_lat",
                "element_corner_2_lat",
                "element_corner_2_lon",
                "element_corner_3_lat",
                "element_corner_3_lon",
            ],
            axis="columns",
        )

    elif geometry_type == "point":
        # Create points from center coordinates
        print("  Building point geometries from center coordinates...")

        def make_point(row):
            return Point(row["lon_center"], row["lat_center"])

        # Create geometry series first to avoid future warning
        geometry_series = df.apply(make_point, axis=1)
        gdf = gpd.GeoDataFrame(df.copy(), geometry=geometry_series)

    else:
        raise ValueError("geometry_type must be 'polygon' or 'point'")

    # Set coordinate system (WGS84)
    gdf.set_crs(epsg=4326, inplace=True)
    print(f"  GeoDataFrame created successfully with CRS: {gdf.crs}")

    return gdf


def save_geo_dataframe(
    # gdf, output_path, filename_base, formats=["shp", "geojson", "gpkg", "parquet"]
    gdf,
    output_path,
    filename_base,
    # formats=["geojson", "gpkg", "parquet"],
    formats=["gpkg", "parquet"],
):
    """
    Save GeoDataFrame in multiple formats including GeoPackage

    Args:
        gdf: GeoDataFrame to save
        output_path: Directory to save files
        filename_base: Base filename (without extension)
        formats: List of formats to save ['shp', 'geojson', 'gpkg', 'parquet']
    """

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    filename_base = filename_base.replace(".parquet", "")  # Remove .parquet if present

    print(f"Saving GeoDataFrame to {len(formats)} formats: {', '.join(formats)}")
    saved_files = []

    # Check for long column names that will be truncated in shapefile
    long_columns = [col for col in gdf.columns if len(col) > 10]
    if long_columns and "shp" in formats:
        print(
            f"  Warning: {len(long_columns)} column names will be truncated in shapefile format"
        )
        print(f"    Longest columns: {long_columns[:3]}...")

    for fmt in formats:
        print(
            f"Planning to make this_output_path from {output_path} and {fmt} with types {type(output_path)} and {type(fmt)}"
        )
        this_output_path = Path(output_path, fmt)
        this_output_path.mkdir(parents=True, exist_ok=True)
        print(f"  Saving {fmt.upper()} format...", end=" ")

        if fmt == "shp":
            filepath = output_path / f"{filename_base}.shp"
            gdf.to_file(filepath)
            saved_files.append(filepath)
            print("✓")

        elif fmt == "geojson":
            filepath = output_path / f"{filename_base}.geojson"
            gdf.to_file(filepath, driver="GeoJSON")
            saved_files.append(filepath)
            print("✓")

        elif fmt == "gpkg":
            filepath = output_path / f"{filename_base}.gpkg"
            gdf.to_file(filepath, driver="GPKG")
            saved_files.append(filepath)
            print("✓")

        elif fmt == "parquet":
            filepath = output_path / f"{filename_base}_geo.parquet"
            gdf.to_parquet(filepath)
            saved_files.append(filepath)
            print("✓")

    print(f"Successfully saved {len(saved_files)} files for {filename_base}")
    return saved_files


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
        data_dict[f"element_corner_{i + 1}_lat"] = ds.lat_node.values[corner_indices]
        data_dict[f"element_corner_{i + 1}_lon"] = ds.lon_node.values[corner_indices]

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

    result_df = compute_grid_resolution(result_df)
    result_df = compute_max_to_mean_ratio(result_df)

    print(f"Created dataframe with {result_df.shape[1]} columns")

    return result_df


def convert_nc_summary_to_parquet(
    config, location_key, create_combined_atlas_output=False
):
    location = config["location_specification"][location_key]
    input_path = file_manager.get_yearly_summary_vap_output_dir(config, location)
    output_path = file_manager.get_vap_summary_parquet_dir(config, location)

    input_nc_files = sorted(list(input_path.rglob("*.nc")))

    cols_for_atlas = ATLAS_COLUMNS.keys()

    dfs = []
    atlas_dfs = []

    for nc_file in input_nc_files:
        print(f"\nProcessing NetCDF file: {nc_file.name}")
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

        print(f"Saving individual parquet: {output_filename}")
        output_df.to_parquet(Path(output_path, output_filename))

        print("Creating individual GeoDataFrame...")
        geo_output_df = create_geo_dataframe(output_df)

        print("OUTPUT_PATH:", output_path)
        save_geo_dataframe(
            geo_output_df, output_path, output_filename.replace(".parquet", "")
        )

        for col in output_df.columns:
            print(f"{col}: {output_df[col].dtype}")

        atlas_df = output_df[
            cols_for_atlas.keys().extend(POLYGON_COLUMNS.keys())
        ].copy()

        atlas_output_path = file_manager.get_vap_atlas_summary_parquet_dir(
            config, location
        )

        atlas_output_filename = (
            file_name_convention_manager.generate_filename_for_data_level(
                output_df,
                location["output_name"],
                config["dataset"]["name"],
                "b5",
                temporal="year_average",
                ext="parquet",
                static_time=date_time_parts,
            )
        )

        print(f"Saving atlas parquet: {atlas_output_filename}")
        atlas_df.to_parquet(Path(atlas_output_path, atlas_output_filename))

        print("Creating atlas GeoDataFrame...")
        geo_atlas_df = create_geo_dataframe(atlas_df)

        save_geo_dataframe(
            geo_atlas_df,
            Path(atlas_output_path, "gis"),
            atlas_output_filename.replace(".parquet", ""),
        )

        # Append to lists for combined output
        dfs.append(output_df)
        atlas_dfs.append(atlas_df)
        print(f"Added to combined datasets (Total files processed: {len(dfs)})")

    if create_combined_atlas_output is True:
        # We need to create complete and atlas datasets for all locations
        # To do this we first find all of the complete summary parquet file for each locations
        # and combine them into a single dataframe
        # Then we save a "full" dataset with all columns to the complete directory and a "atlas" dataset with only the atlas columns to the "atlas_subset" directory

        print("\n=== CREATING COMBINED OUTPUTS ===")
        # Calculate number of locations dynamically
        num_locations = len(config["location_specification"])
        print(f"Creating atlas GIS outputs for {num_locations} locations")

        summary_dfs = []

        for location_key, location in config["location_specification"].items():
            summary_parquet_output_path = file_manager.get_vap_summary_parquet_dir(
                config, location
            )

            # We put the gis files is a gis subdirectory, so this should only pick up the parquet file
            # This is easier in pandas than trying to combine gis files
            complete_summary_parquet = sorted(
                list(summary_parquet_output_path.glob("*.parquet"))
            )[0]

            print(f"Processing atlas file: {complete_summary_parquet.name}")
            this_summary_df = pd.read_parquet(complete_summary_parquet)
            this_summary_df["name"] = location["output_name"]
            this_summary_df["label"] = location["label"]
            this_summary_df["location_face_index"] = this_summary_df.index
            summary_dfs.append(this_summary_df)

        # Combine all dataframes
        combined_df = pd.concat(summary_dfs, ignore_index=True)
        combined_df = combined_df.reset_index(drop=True)

        print(
            f"Combined complete dataset: {combined_df.shape[0]} rows, {combined_df.shape[1]} columns"
        )

        geo_combined_df = create_geo_dataframe(combined_df, geometry_type="polygon")

        geo_atlas_df = geo_combined_df[[ATLAS_COLUMNS.keys()]]

        # Now we need to save the complete dataset

        this_output_path = file_manager.get_combined_vap_atlas(config, location)
        this_complete_output_path = Path("complete", this_output_path)
        this_atlas_output_path = Path("atlas_subset", this_output_path)

        complete_file_name = (
            file_name_convention_manager.generate_filename_for_data_level(
                combined_df,
                f"all_columns_{num_locations}_tidal_locations_atlas",
                config["dataset"]["name"],
                "b7",
                temporal="year_average",
                ext=None,
                version=config["dataset"]["gis_output_version"],
                include_creation_timestamp=True,
                include_dataset_time=False,
            )
        )

        atlas_file_name = file_name_convention_manager.generate_filename_for_data_level(
            combined_df,
            f"atlas_subset_{num_locations}_tidal_locations_atlas",
            config["dataset"]["name"],
            "b7",
            temporal="year_average",
            ext=None,
            version=config["dataset"]["gis_output_version"],
            include_creation_timestamp=True,
            include_dataset_time=False,
        )

        # Archive existing files in complete directory before creating new ones
        complete_archive_dir = this_complete_output_path / "archive"
        if this_complete_output_path.exists():
            existing_complete_files = [
                f
                for f in this_complete_output_path.rglob("*")
                if f.is_file() and "archive" not in f.parts
            ]
            if existing_complete_files:
                complete_archive_dir.mkdir(parents=True, exist_ok=True)
                print(
                    f"Archiving {len(existing_complete_files)} files from {this_complete_output_path}"
                )
                for file_path in existing_complete_files:
                    rel_path = file_path.relative_to(this_complete_output_path)
                    archive_path = complete_archive_dir / rel_path
                    archive_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(file_path, archive_path)

        # Create and save complete GIS outputs (all columns)
        print("\n=== CREATING COMPLETE GIS OUTPUTS ===")
        print(f"Processing complete dataset with {combined_df.shape[1]} columns...")
        save_geo_dataframe(
            geo_combined_df,
            this_complete_output_path,
            complete_file_name,
            formats=["geojson", "gpkg", "parquet"],
        )

        # Archive existing files in atlas_subset directory before creating new ones
        atlas_archive_dir = this_atlas_output_path / "archive"
        if this_atlas_output_path.exists():
            existing_atlas_subset_files = [
                f
                for f in this_atlas_output_path.rglob("*")
                if f.is_file() and "archive" not in f.parts
            ]
            if existing_atlas_subset_files:
                atlas_archive_dir.mkdir(parents=True, exist_ok=True)
                print(
                    f"Archiving {len(existing_atlas_subset_files)} files from {this_atlas_output_path}"
                )
                for file_path in existing_atlas_subset_files:
                    rel_path = file_path.relative_to(this_atlas_output_path)
                    archive_path = atlas_archive_dir / rel_path
                    archive_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(file_path, archive_path)

        # Create and save atlas subset GIS outputs (atlas columns only)
        print("\n=== CREATING ATLAS SUBSET GIS OUTPUTS ===")
        print(f"Processing atlas subset with {geo_atlas_df.shape[1]} columns...")

        save_geo_dataframe(
            geo_atlas_df,
            this_atlas_output_path,
            atlas_file_name,
            formats=["geojson", "gpkg", "parquet"],
        )

        print("\n=== PROCESSING SUMMARY ===")
        print(
            f"✓ Complete dataset: {geo_combined_df.shape[0]} rows, {geo_combined_df.shape[1]} columns"
        )
        print(
            f"✓ Atlas subset: {geo_atlas_df.shape[0]} rows, {geo_atlas_df.shape[1]} columns"
        )
        print(f"✓ Number of locations processed: {num_locations}")
        print(f"✓ Files processed per location: {len(dfs)}")
        print(f"✓ Combined Output directory: {this_complete_output_path}")
        print(f"✓ Atlas Output directory: {this_atlas_output_path}")

    # return combined_df, combined_atlas_df
