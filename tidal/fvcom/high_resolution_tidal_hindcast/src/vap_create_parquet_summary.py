from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

from shapely.geometry import Point, Polygon

from . import file_manager, file_name_convention_manager


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

    # Grid resolution as average edge length
    df["grid_resolution_meters"] = (edge1 + edge2 + edge3) / 3

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
    gdf, output_path, filename_base, formats=["shp", "geojson", "gpkg", "parquet"]
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
        print(f"  Saving {fmt.upper()} format...", end=" ")

        if fmt == "shp":
            filepath = output_path / f"{filename_base}.shp"
            # Suppress the column name warning since we already warned about it
            import warnings

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message="Column names longer than 10 characters"
                )
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


def convert_nc_summary_to_parquet(config, location_key):
    location = config["location_specification"][location_key]
    input_path = file_manager.get_yearly_summary_vap_output_dir(config, location)
    output_path = file_manager.get_vap_summary_parquet_dir(config, location)

    input_nc_files = sorted(list(input_path.rglob("*.nc")))

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
        "vap_water_column_sea_water_speed_max_to_mean_ratio",
        "vap_water_column_mean_sea_water_power_density",
        "vap_water_column_95th_percentile_sea_water_power_density",
        "vap_water_column_sea_water_power_density_max_to_mean_ratio",
        "vap_sea_floor_depth",
        "grid_resolution_meters",
    ]

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
        save_geo_dataframe(
            geo_output_df, output_path, output_filename.replace(".parquet", "")
        )

        for col in output_df.columns:
            print(f"{col}: {output_df[col].dtype}")

        atlas_df = output_df[cols_for_atlas]

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
            atlas_output_path,
            atlas_output_filename.replace(".parquet", ""),
        )

        # Append to lists for combined output
        dfs.append(output_df)
        atlas_dfs.append(atlas_df)
        print(f"Added to combined datasets (Total files processed: {len(dfs)})")

    print("\n=== CREATING COMBINED OUTPUTS ===")
    # Calculate number of locations dynamically
    num_locations = len(config["location_specification"])
    print(f"Processing {num_locations} total locations in configuration")

    # Create combined outputs
    combined_output_path = file_manager.get_combined_vap_atlas(config, location)

    # Combine all dataframes
    print("Concatenating all individual DataFrames...")
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_atlas_df = pd.concat(atlas_dfs, ignore_index=True)
    print(
        f"Combined complete dataset: {combined_df.shape[0]} rows, {combined_df.shape[1]} columns"
    )
    print(
        f"Combined atlas dataset: {combined_atlas_df.shape[0]} rows, {combined_atlas_df.shape[1]} columns"
    )

    # Generate combined filenames
    combined_output_filename = (
        file_name_convention_manager.generate_filename_for_data_level(
            combined_df,
            f"all_{num_locations}_tidal_locations",
            config["dataset"]["name"],
            "b7",
            temporal="year_average",
            ext=None,
        )
    )

    combined_atlas_filename = (
        file_name_convention_manager.generate_filename_for_data_level(
            combined_atlas_df,
            f"all_{num_locations}_tidal_locations_atlas",
            config["dataset"]["name"],
            "b7",
            temporal="year_average",
            ext=None,
        )
    )

    # Save combined parquet files
    print(f"\nSaving combined parquet files to: {combined_output_path}")
    combined_df.to_parquet(
        Path(combined_output_path, f"{combined_output_filename}.parquet")
    )
    combined_atlas_df.to_parquet(
        Path(combined_output_path, f"{combined_atlas_filename}.parquet")
    )
    print("✓ Combined parquet files saved")

    # Create and save complete GIS outputs (all columns)
    print("\n=== CREATING COMPLETE GIS OUTPUTS ===")
    print(f"Processing complete dataset with {combined_df.shape[1]} columns...")
    geo_combined_df = create_geo_dataframe(combined_df, geometry_type="polygon")
    save_geo_dataframe(
        geo_combined_df,
        combined_output_path,
        f"{combined_output_filename}_complete",
        formats=["shp", "geojson", "gpkg", "parquet"],
    )

    # Create and save atlas subset GIS outputs (atlas columns only)
    print("\n=== CREATING ATLAS SUBSET GIS OUTPUTS ===")
    print(f"Processing atlas subset with {combined_atlas_df.shape[1]} columns...")
    geo_combined_atlas_df = create_geo_dataframe(
        combined_atlas_df, geometry_type="polygon"
    )
    save_geo_dataframe(
        geo_combined_atlas_df,
        combined_output_path,
        f"{combined_atlas_filename}_atlas_subset",
        formats=["shp", "geojson", "gpkg", "parquet"],
    )

    print("\n=== PROCESSING SUMMARY ===")
    print(
        f"✓ Complete dataset: {combined_df.shape[0]} rows, {combined_df.shape[1]} columns"
    )
    print(
        f"✓ Atlas subset: {combined_atlas_df.shape[0]} rows, {combined_atlas_df.shape[1]} columns"
    )
    print(f"✓ Number of locations processed: {num_locations}")
    print(f"✓ Files processed per location: {len(dfs)}")
    print(f"✓ Output directory: {combined_output_path}")

    return combined_df, combined_atlas_df
