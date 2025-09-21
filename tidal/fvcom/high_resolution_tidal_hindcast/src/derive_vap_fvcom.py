import gc
import time
import pytz

from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import pyproj
from shapely.geometry import Point
from shapely.ops import nearest_points
from timezonefinder import TimezoneFinder

from . import attrs_manager, file_manager, file_name_convention_manager, nc_manager
from .distance_to_shore_manager import DistanceToShoreCalculator
from .jurisdiction_manager import JurisdictionCalculator

output_names = {
    # Original Data
    "u": "u",
    "v": "v",
    "h_center": "h_center",
    # VAP Data
    "speed": "vap_sea_water_speed",
    "to_direction": "vap_sea_water_to_direction",
    "from_direction": "vap_sea_water_from_direction",
    "power_density": "vap_sea_water_power_density",
    "element_volume": "vap_element_volume",
    "volume_flux": "vap_volume_flux",
    "volume_flux_average": "vap_water_column_volume_mean_flux",
    "zeta_center": "vap_zeta_center",
    "surface_elevation": "vap_surface_elevation",
    "depth": "vap_sigma_depth",
    "sea_floor_depth": "vap_sea_floor_depth",
    "mean": "vap_water_column_mean",
    "median": "vap_water_column_median",
    "max": "vap_water_column_max",
    "p95": "vap_water_column_<PERCENTILE>th_percentile",
    "utc_timezone_offset": "vap_utc_timezone_offset",
    "distance_to_shore": "vap_distance_to_shore",
    "jurisdiction": "vap_jurisdiction",
}


def get_face_center_precalculations_path(config, location):
    """Get path to consolidated face center precalculations parquet file"""
    tracking_path = file_manager.get_tracking_output_dir(config, location)
    location_config = (
        config["location_specification"][location]
        if isinstance(location, str)
        else location
    )
    return Path(
        tracking_path,
        f"{location_config['output_name']}_face_center_precalculations.parquet",
    )


def _load_and_validate_existing_precalculations(config, location_key):
    """
    Load existing precalculations and validate coordinates and completeness.

    Returns
    -------
    tuple[pd.DataFrame or None, list[str]]
        Tuple of (existing_dataframe, missing_columns). If validation fails,
        returns (None, []) indicating full recalculation is needed.
    """
    location = config["location_specification"][location_key]
    parquet_path = get_face_center_precalculations_path(config, location)

    # Check if file exists
    if not parquet_path.exists():
        print(f"Precalculations file does not exist: {parquet_path}")
        return None, []

    try:
        # Load existing data
        print(f"Loading existing precalculations from: {parquet_path}")
        existing_df = pd.read_parquet(parquet_path)
        print(
            f"Loaded existing data: {len(existing_df)} faces, {len(existing_df.columns)} columns"
        )

        # Expected columns (all that should be calculated)
        expected_columns = [
            "latitude_center",
            "longitude_center",
            "timezone_offset",
            "distance_to_shore",
            "jurisdiction",
            "closest_country",
            "closest_state_province",
            "mean_navd88_offset",
        ]

        # Check which columns are missing or have null values
        missing_columns = []
        for col in expected_columns:
            if col not in existing_df.columns:
                missing_columns.append(col)
                print(f"Missing column: {col}")
            elif existing_df[col].isnull().any():
                missing_columns.append(col)
                print(f"Column {col} has null values")

        # Validate coordinates against reference data if lat/lon are present
        if (
            "latitude_center" in existing_df.columns
            and "longitude_center" in existing_df.columns
        ):
            print("Validating coordinates against reference data...")
            try:
                reference_df = _initialize_face_coordinates_dataframe(
                    config, location_key
                )

                # Check face count consistency
                if len(existing_df) != len(reference_df):
                    print(
                        f"Face count mismatch: existing={len(existing_df)}, reference={len(reference_df)}"
                    )
                    return None, []

                # Check coordinate consistency with same tolerance as original validation
                if not np.allclose(
                    existing_df["longitude_center"],
                    reference_df["longitude_center"],
                    rtol=1e-10,
                ):
                    print("Longitude coordinates do not match reference data")
                    return None, []

                if not np.allclose(
                    existing_df["latitude_center"],
                    reference_df["latitude_center"],
                    rtol=1e-10,
                ):
                    print("Latitude coordinates do not match reference data")
                    return None, []

                print("Coordinate validation successful")

            except Exception as e:
                print(f"Warning: Could not validate coordinates against reference: {e}")
                print("Proceeding with full recalculation...")
                return None, []
        else:
            # If coordinates are missing, need full recalculation
            missing_columns.extend(["latitude_center", "longitude_center"])

        return existing_df, missing_columns

    except Exception as e:
        print(f"Warning: Could not load or validate existing precalculations: {e}")
        print("Proceeding with full recalculation...")
        return None, []


def calculate_and_save_face_center_precalculations(
    config, location_key, skip_if_precalculated=False
):
    """
    Calculate and save all face-centered precalculated data in correct dependency order.

    This function creates a consolidated DataFrame with face-centered spatial data and
    also calculates the NAVD88 offset data. Both are saved for later use in VAP processing.

    Parameters
    ----------
    config : dict
        Configuration dictionary
    location_key : str
        Location key from config
    skip_if_precalculated : bool, default False
        If True, check if precalculations already exist and skip recalculation if
        coordinates match and all columns are present with non-null values

    Returns
    -------
    Path
        Path to the parquet file containing precalculations

    Raises
    ------
    ValueError
        If files have inconsistent face coordinates or calculations fail
    FileNotFoundError
        If no NC files found in the specified path
    """
    location = config["location_specification"][location_key]
    location_name = location["label"]

    print(f"Calculating all face-centered precalculations for {location_name}...")

    try:
        # Check if we should skip based on existing precalculations
        if skip_if_precalculated:
            existing_df, missing_columns = _load_and_validate_existing_precalculations(
                config, location_key
            )
            if existing_df is not None:
                if not missing_columns:
                    print(
                        "All precalculations already exist and are valid. Skipping recalculation."
                    )
                    parquet_path = get_face_center_precalculations_path(
                        config, location
                    )
                    return parquet_path
                else:
                    print(
                        f"Found existing precalculations but missing columns: {missing_columns}"
                    )
                    print("Will recalculate only missing columns...")
                    df = existing_df
            else:
                print(
                    "No valid existing precalculations found. Performing full calculation..."
                )
                df = None
        else:
            df = None

        # Step 1: Initialize DataFrame with face coordinates (if not loaded from existing)
        if df is None:
            print("Step 1: Initializing face coordinates DataFrame...")
            df = _initialize_face_coordinates_dataframe(config, location_key)
            missing_columns = [
                "timezone_offset",
                "distance_to_shore",
                "jurisdiction",
                "closest_country",
                "closest_state_province",
                "mean_navd88_offset",
            ]

        # Step 2: Add timezone offset data (if missing)
        if "timezone_offset" in missing_columns:
            print("Step 2: Adding timezone offset data...")
            df = _add_timezone_offset_to_dataframe(df, config, location_key)

        # # Step 3: Add distance to shore data (if missing)
        # if "distance_to_shore" in missing_columns:
        #     print("Step 3: Adding distance to shore data...")
        #     df = _add_distance_to_shore_to_dataframe(df, config, location_key)

        # Step 4: Add jurisdiction data (if missing, depends on distance to shore)
        jurisdiction_columns = [
            "jurisdiction",
            "closest_country",
            "closest_state_province",
        ]
        if any(col in missing_columns for col in jurisdiction_columns):
            print("Step 4: Adding jurisdiction data...")
            df = _add_jurisdiction_to_dataframe(df, config, location_key)

        # Step 5: Calculate NAVD88 offset (if missing)
        if "mean_navd88_offset" in missing_columns:
            print("Step 5: Calculating mean NAVD88 offset...")
            df = _add_navd88_offset_to_dataframe(df, config, location_key)

        # Step 6: Save consolidated parquet file
        print("Step 6: Saving consolidated precalculations...")
        parquet_path = _save_face_precalculations_dataframe(df, config, location_key)

        print("All face-centered precalculations complete!")
        print(f"  Parquet file: {parquet_path}")

        return parquet_path

    except Exception as e:
        print(f"ERROR: Face-centered precalculations failed: {e}")
        raise


def _initialize_face_coordinates_dataframe(config, location_key):
    """Create initial DataFrame with face_index, lat_center, lon_center"""
    location = config["location_specification"][location_key]
    input_path = file_manager.get_standardized_partition_output_dir(config, location)

    # Find all NC files for coordinate validation
    nc_files = sorted(list(input_path.rglob("*.nc")))
    if not nc_files:
        raise FileNotFoundError(f"No NetCDF files found in {input_path}")

    print(f"Initializing coordinates from {len(nc_files)} files...")

    # Get reference coordinates from first file
    reference_coords = None

    for i, nc_file in enumerate(nc_files):
        print(
            f"Validating coordinates from file {i + 1}/{len(nc_files)}: {nc_file.name}"
        )

        with xr.open_dataset(nc_file) as ds:
            current_coords = {
                "lon_center": ds.lon_center.values,
                "lat_center": ds.lat_center.values,
                "n_faces": len(ds.face),
                "face_index": ds.face.values,
            }

            if reference_coords is None:
                reference_coords = current_coords
                print(f"Reference coordinates set: {current_coords['n_faces']} faces")
            else:
                # Validate consistency (same as NAVD88 function)
                if current_coords["n_faces"] != reference_coords["n_faces"]:
                    raise ValueError(
                        f"File {nc_file.name} has {current_coords['n_faces']} faces, "
                        f"but reference has {reference_coords['n_faces']} faces"
                    )

                if not np.allclose(
                    current_coords["lon_center"],
                    reference_coords["lon_center"],
                    rtol=1e-10,
                ):
                    raise ValueError(
                        f"File {nc_file.name} has inconsistent lon_center coordinates"
                    )

                if not np.allclose(
                    current_coords["lat_center"],
                    reference_coords["lat_center"],
                    rtol=1e-10,
                ):
                    raise ValueError(
                        f"File {nc_file.name} has inconsistent lat_center coordinates"
                    )

    # Create DataFrame with face_index as index
    df = pd.DataFrame(
        {
            "latitude_center": reference_coords["lat_center"].astype(np.float64),
            "longitude_center": reference_coords["lon_center"].astype(np.float64),
        },
        index=reference_coords["face_index"],
    )

    df.index.name = "face_index"

    print(f"Initialized DataFrame: {len(df)} faces, columns: {list(df.columns)}")
    return df


def _add_timezone_offset_to_dataframe(df, config, location_key):
    """Add timezone_offset column to existing DataFrame"""

    print(f"Calculating timezone offsets for {len(df)} face coordinates...")

    tf = TimezoneFinder()
    timezone_offsets = []

    for i, (face_idx, row) in enumerate(df.iterrows()):
        if i % 1000 == 0:
            print(f"  Processing face {i + 1}/{len(df)}")

        lat = row["latitude_center"]
        lon = row["longitude_center"]

        # Get timezone string for this coordinate
        timezone_str = tf.timezone_at(lat=lat, lng=lon)

        tz = pytz.timezone(timezone_str)
        # Get UTC offset in hours (using a reference date to avoid DST issues)
        reference_date = pd.Timestamp("2023-01-15 12:00:00")  # Winter time
        localized_dt = tz.localize(reference_date)
        utc_offset_hours = int(localized_dt.utcoffset().total_seconds() / 3600)
        timezone_offsets.append(utc_offset_hours)

    df["timezone_offset"] = pd.Series(timezone_offsets, dtype=np.int16, index=df.index)

    # Save metadata JSON
    metadata = {
        "variable_name": "timezone_offset",
        "standard_name": "utc_timezone_offset",
        "long_name": "UTC Timezone Offset",
        "units": "hours",
        "description": "UTC timezone offset in hours for each face center coordinate",
        "computation": "TimezoneFinder library with pytz timezone database",
        "reference_date": "2023-01-15 (winter time to avoid DST)",
        "default_value": 0,
        "dtype": "int16",
        "unique_values": sorted(df["timezone_offset"].unique().tolist()),
        "value_counts": df["timezone_offset"].value_counts().to_dict(),
        "creation_date": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
    }

    _save_metadata_json(metadata, config, location_key, "timezone_offset")

    print(
        f"Added timezone_offset column. Unique values: {sorted(df['timezone_offset'].unique())}"
    )
    return df


def _add_distance_to_shore_to_dataframe(df, config, location_key):
    """Add distance_to_shore column to existing DataFrame using GSHHG data"""

    print(f"Calculating distance to shore for {len(df)} face coordinates...")

    # Initialize the distance calculator with nautical miles (for compatibility)
    distance_calculator = DistanceToShoreCalculator(config, units="nautical_miles")

    # Calculate distance to shore
    df_with_distance = distance_calculator.calc_distance_to_shore(df)

    # Save metadata JSON
    metadata = distance_calculator.get_metadata()

    # Add statistics to metadata
    metadata["statistics"] = {
        "min": float(df_with_distance["distance_to_shore"].min()),
        "max": float(df_with_distance["distance_to_shore"].max()),
        "mean": float(df_with_distance["distance_to_shore"].mean()),
        "median": float(df_with_distance["distance_to_shore"].median()),
    }

    _save_metadata_json(metadata, config, location_key, "distance_to_shore")

    print(
        f"Added distance_to_shore column and closest shore coordinates. Stats: min={df_with_distance['distance_to_shore'].min():.2f}, max={df_with_distance['distance_to_shore'].max():.2f}, mean={df_with_distance['distance_to_shore'].mean():.2f} NM"
    )
    return df_with_distance


def _calculate_closest_admin_boundaries(df, config):
    """
    Calculate closest country and state/province for each face center coordinate using Natural Earth administrative boundary data.

    This function requires accurate Natural Earth administrative boundary data and will fail if the required
    configuration paths are not provided. No fallback or approximation methods are used to ensure data accuracy.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with face center coordinates indexed by face_index
    config : dict
        Configuration dictionary that must contain:
        - 'natural_earth_countries_data_path': Path to Natural Earth countries shapefile
        - 'natural_earth_states_data_path': Path to Natural Earth states/provinces shapefile

    Returns
    -------
    tuple[list[str], list[str]]
        Tuple containing (closest_countries, closest_states) lists

    Raises
    ------
    ValueError
        If required Natural Earth data paths are not provided in config
        If no administrative boundaries are found for any coordinate
        If column names cannot be determined from Natural Earth data
    FileNotFoundError
        If Natural Earth data files cannot be loaded
    """

    # Get administrative data paths from config
    countries_path = config.get("natural_earth_countries_data_path")
    states_path = config.get("natural_earth_states_data_path")

    if not countries_path:
        raise ValueError(
            "Config must contain 'natural_earth_countries_data_path' for closest country calculation. "
            "Accurate administrative boundary data is required."
        )

    # Load country boundaries
    print("Loading country boundary data...")
    countries_gdf = gpd.read_file(countries_path)
    print(f"Loaded country data: {len(countries_gdf)} features")

    # Load state/province boundaries - required for accurate state/province data
    if not states_path:
        raise ValueError(
            "Config must contain 'natural_earth_states_data_path' for closest state/province calculation. "
            "Accurate administrative boundary data is required."
        )

    print("Loading state/province boundary data...")
    states_gdf = gpd.read_file(states_path)
    print(f"Loaded state/province data: {len(states_gdf)} features")

    # Require closest shore point data for accurate admin boundary calculation
    if "closest_shore_lat" not in df.columns or "closest_shore_lon" not in df.columns:
        raise ValueError(
            "DataFrame must contain 'closest_shore_lat' and 'closest_shore_lon' columns. "
            "Please run _add_distance_to_shore_to_dataframe() first to generate closest shore point data."
        )

    print("Using closest shore point coordinates for admin boundary calculation...")
    # Use closest shore point coordinates, but fall back to face center for on-land points (distance = 0)
    lats = []
    lons = []
    for _, row in df.iterrows():
        if row.get("distance_to_shore", float("inf")) == 0.0:
            # For points on land, use face center coordinates
            lats.append(row["latitude_center"])
            lons.append(row["longitude_center"])
        else:
            # For points in water, use closest shore point coordinates
            lats.append(row["closest_shore_lat"])
            lons.append(row["closest_shore_lon"])

    geometry = [Point(lon, lat) for lat, lon in zip(lats, lons)]
    points_gdf = gpd.GeoDataFrame(
        df[["latitude_center", "longitude_center"]], geometry=geometry, crs="EPSG:4326"
    )
    on_land_count = sum(
        1
        for i, row in df.iterrows()
        if row.get("distance_to_shore", float("inf")) == 0.0
    )
    water_count = len(df) - on_land_count
    print(
        f"Using closest shore points for {water_count} faces, face center for {on_land_count} on-land faces"
    )

    # Transform to appropriate projection for distance calculations (EPSG:4087 - World Equidistant Cylindrical)
    points_projected = points_gdf.to_crs("EPSG:4087")
    countries_projected = countries_gdf.to_crs("EPSG:4087")
    states_projected = states_gdf.to_crs("EPSG:4087")

    # Create spatial indexes for efficiency
    countries_sindex = countries_projected.sindex
    states_sindex = states_projected.sindex

    closest_countries = []
    closest_states = []
    batch_size = 1000

    for batch_start in range(0, len(points_projected), batch_size):
        batch_end = min(batch_start + batch_size, len(points_projected))
        print(
            f"  Processing admin boundary batch {batch_start + 1}-{batch_end} of {len(points_projected)}"
        )

        batch_points = points_projected.iloc[batch_start:batch_end]

        for idx, point_row in batch_points.iterrows():
            point_geom = point_row.geometry

            # Find closest country
            possible_countries_index = list(
                countries_sindex.intersection(point_geom.bounds)
            )
            if len(possible_countries_index) == 0:
                # Expand search area if no immediate matches
                buffer = point_geom.buffer(100000)  # 100km buffer
                possible_countries_index = list(
                    countries_sindex.intersection(buffer.bounds)
                )

            if len(possible_countries_index) > 0:
                possible_countries = countries_projected.iloc[possible_countries_index]
                distances = possible_countries.geometry.distance(point_geom)
                min_idx = distances.idxmin()
                # Try different common column names for country names
                for col in ["NAME", "ADMIN", "name", "Country"]:
                    if col in countries_projected.columns:
                        country_name = countries_projected.loc[min_idx, col]
                        break
                else:
                    raise ValueError(
                        f"Could not find a valid column name for country data in Natural Earth dataset. "
                        f"Available columns: {list(countries_projected.columns)}"
                    )
                closest_countries.append(country_name)
            else:
                raise ValueError(
                    f"No country features found near point at index {idx} "
                    f"({point_row['latitude_center']:.4f}, {point_row['longitude_center']:.4f}) "
                    f"even after expanding search to 100km buffer. This suggests incomplete "
                    f"country boundary data coverage."
                )

            # Find closest state/province
            possible_states_index = list(states_sindex.intersection(point_geom.bounds))
            if len(possible_states_index) == 0:
                # Expand search area if no immediate matches
                buffer = point_geom.buffer(100000)  # 100km buffer
                possible_states_index = list(states_sindex.intersection(buffer.bounds))

            if len(possible_states_index) > 0:
                possible_states = states_projected.iloc[possible_states_index]
                distances = possible_states.geometry.distance(point_geom)
                min_idx = distances.idxmin()
                # Try different common column names for state/province names
                for col in ["NAME", "ADMIN", "NAME_1", "name", "State", "Province"]:
                    if col in states_projected.columns:
                        state_name = states_projected.loc[min_idx, col]
                        break
                else:
                    raise ValueError(
                        f"Could not find a valid column name for state/province data in Natural Earth dataset. "
                        f"Available columns: {list(states_projected.columns)}"
                    )
                closest_states.append(state_name)
            else:
                raise ValueError(
                    f"No state/province features found near point at index {idx} "
                    f"({point_row['latitude_center']:.4f}, {point_row['longitude_center']:.4f}) "
                    f"even after expanding search to 100km buffer. This suggests incomplete "
                    f"state/province boundary data coverage."
                )

    return closest_countries, closest_states


def _add_jurisdiction_to_dataframe(df, config, location_key):
    """Add jurisdiction column to existing DataFrame using NOAA jurisdiction data"""

    print(f"Calculating maritime jurisdiction for {len(df)} faces...")

    # Add location_key to config for jurisdiction calculation
    config_with_location = config.copy()
    config_with_location["location_key"] = location_key

    # Initialize the jurisdiction calculator
    jurisdiction_calculator = JurisdictionCalculator(config_with_location)

    # Calculate jurisdiction
    df_with_jurisdiction = jurisdiction_calculator.calc_jurisdiction(df)

    # Save metadata JSON for jurisdiction
    metadata = jurisdiction_calculator.get_metadata()

    # Calculate statistics for metadata
    unique_jurisdictions, counts = np.unique(
        df_with_jurisdiction["jurisdiction"], return_counts=True
    )
    jurisdiction_stats = dict(zip(unique_jurisdictions, counts.tolist()))

    metadata["unique_values"] = unique_jurisdictions.tolist()
    metadata["value_counts"] = jurisdiction_stats

    _save_metadata_json(metadata, config, location_key, "jurisdiction")

    return df_with_jurisdiction


def _add_navd88_offset_to_dataframe(df, config, location_key):
    """Add mean_navd88_offset column to existing DataFrame"""
    print(f"Calculating mean NAVD88 offset for {len(df)} faces...")

    location = config["location_specification"][location_key]
    input_path = file_manager.get_standardized_partition_output_dir(config, location)

    # Find all NC files
    nc_files = sorted(list(input_path.rglob("*.nc")))
    if not nc_files:
        raise FileNotFoundError(f"No NetCDF files found in {input_path}")

    zeta_center_data = []

    for i, nc_file in enumerate(nc_files):
        if i % 5 == 0:
            print(f"  Processing file {i + 1}/{len(nc_files)}: {nc_file.name}")

        with xr.open_dataset(nc_file) as ds:
            # Calculate zeta_center for this file
            ds_with_zeta = calculate_zeta_center(ds)

            # Extract zeta_center data
            zeta_center = ds_with_zeta[output_names["zeta_center"]]

            # Validate that face indices match our DataFrame
            if not np.array_equal(zeta_center.face.values, df.index.values):
                raise ValueError(
                    f"File {nc_file.name} has inconsistent face indices with DataFrame"
                )

            # Store the zeta_center data (face dimension should match DataFrame index)
            zeta_center_data.append(zeta_center.values)

    # Concatenate all data along time axis
    print("Concatenating data and calculating temporal mean...")
    all_zeta_data = np.concatenate(zeta_center_data, axis=0)  # (total_time, n_faces)

    # Calculate mean across all time steps for each face
    mean_navd88_offset = np.mean(all_zeta_data, axis=0)  # (n_faces,)

    # Add to DataFrame with same index
    df["mean_navd88_offset"] = pd.Series(
        mean_navd88_offset, dtype=np.float32, index=df.index
    )

    # Save metadata JSON
    metadata = {
        "variable_name": "mean_navd88_offset",
        "standard_name": "mean_sea_surface_height_offset_above_geoid",
        "long_name": "Mean NAVD88 Offset for Surface Elevation Calculation",
        "units": "m",
        "description": (
            "Temporal mean of zeta_center values across all input files. "
            "This offset represents the mean deviation from NAVD88 datum "
            "and should be subtracted from instantaneous zeta_center values "
            "to obtain surface elevation relative to mean sea level."
        ),
        "computation": "mean(zeta_center) across all timesteps and files",
        "input_files_count": len(nc_files),
        "total_timesteps": all_zeta_data.shape[0],
        "coordinate_reference": "NAVD88",
        "dtype": "float32",
        "statistics": {
            "min": float(df["mean_navd88_offset"].min()),
            "max": float(df["mean_navd88_offset"].max()),
            "mean": float(df["mean_navd88_offset"].mean()),
            "std": float(df["mean_navd88_offset"].std()),
        },
        "creation_date": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
    }

    _save_metadata_json(metadata, config, location_key, "mean_navd88_offset")

    print(
        f"Added mean_navd88_offset column. Stats: min={df['mean_navd88_offset'].min():.3f}, max={df['mean_navd88_offset'].max():.3f}, mean={df['mean_navd88_offset'].mean():.3f} m"
    )
    return df


def _save_face_precalculations_dataframe(df, config, location_key):
    """Save consolidated DataFrame to parquet file"""
    location = config["location_specification"][location_key]
    output_path = get_face_center_precalculations_path(config, location)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save to parquet with optimal settings
    df.to_parquet(
        output_path,
        index=True,  # Include face_index
        compression="snappy",
        engine="pyarrow",
    )

    print(f"Consolidated DataFrame saved: {len(df)} faces, {len(df.columns)} columns")
    print(f"Columns: {list(df.columns)}")
    print(f"File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

    return output_path


def _save_metadata_json(metadata, config, location_key, variable_name):
    """Save metadata JSON file"""
    location = config["location_specification"][location_key]
    output_path = file_manager.get_tracking_output_dir(config, location)
    json_path = Path(
        output_path, f"{location['output_name']}_{variable_name}_metadata.json"
    )

    # Ensure output directory exists
    json_path.parent.mkdir(parents=True, exist_ok=True)

    import json

    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Metadata saved: {json_path}")


def _load_precomputed_face_data(ds, config, location_key=None):
    """
    Load precomputed face-centered data from parquet file and validate consistency with dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing lat_center, lon_center, and face coordinates
    config : dict
        Configuration dictionary
    location_key : str, optional
        Location key to use. If None, will attempt to determine from config context

    Returns
    -------
    pandas.DataFrame
        DataFrame with precomputed face data indexed by face_index containing:
        - latitude_center, longitude_center
        - timezone_offset
        - distance_to_shore
        - closest_shore_lat, closest_shore_lon
        - jurisdiction
        - closest_country
        - closest_state_province
        - mean_navd88_offset

    Raises
    ------
    ValueError
        If precomputed data file doesn't exist or validation fails
    FileNotFoundError
        If parquet file not found
    """
    # Determine location_key if not provided
    if location_key is None:
        # Try to determine from config context - look for single location
        location_specs = config.get("location_specification", {})
        if len(location_specs) == 1:
            location_key = list(location_specs.keys())[0]
        else:
            raise ValueError(
                "Cannot determine location_key automatically. Multiple or no locations in config. "
                "Please provide location_key parameter explicitly."
            )

    # Get location configuration
    location_config = config["location_specification"][location_key]

    # Get parquet file path
    parquet_path = get_face_center_precalculations_path(config, location_config)

    # Check if file exists
    if not parquet_path.exists():
        raise ValueError(
            f"Precomputed face data not found at {parquet_path}. "
            f"Please run calculate_and_save_face_center_precalculations(config, '{location_key}') first."
        )

    # Load the DataFrame
    print(f"Loading precomputed face data from: {parquet_path}")
    df = pd.read_parquet(parquet_path)

    # Validate that the DataFrame has expected columns
    required_columns = [
        "latitude_center",
        "longitude_center",
        "timezone_offset",
        "distance_to_shore",
        "closest_shore_lat",
        "closest_shore_lon",
        "jurisdiction",
        "closest_country",
        "closest_state_province",
        "mean_navd88_offset",
    ]

    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Precomputed data missing required columns: {missing_columns}. "
            f"Available columns: {list(df.columns)}"
        )

    # Validate consistency with dataset
    ds_face_indices = ds.face.values
    ds_lat_centers = ds.lat_center.values
    ds_lon_centers = ds.lon_center.values

    # Check that face indices match
    if not np.array_equal(ds_face_indices, df.index.values):
        raise ValueError(
            f"Face indices mismatch between dataset and precomputed data. "
            f"Dataset faces: {len(ds_face_indices)}, DataFrame faces: {len(df)}"
        )

    # Check coordinate consistency (using the same tolerance as validation functions)
    if not np.allclose(ds_lat_centers, df["latitude_center"].values, rtol=1e-10):
        raise ValueError(
            "Latitude coordinates mismatch between dataset and precomputed data"
        )

    if not np.allclose(ds_lon_centers, df["longitude_center"].values, rtol=1e-10):
        raise ValueError(
            "Longitude coordinates mismatch between dataset and precomputed data"
        )

    print(f"Successfully loaded and validated precomputed data for {len(df)} faces")
    print("Available precomputed values:")
    print(f"  - Timezone offsets: {sorted(df['timezone_offset'].unique())}")
    print(
        f"  - Distance to shore range: {df['distance_to_shore'].min():.2f} - {df['distance_to_shore'].max():.2f} NM"
    )
    print(f"  - Jurisdictions: {len(df['jurisdiction'].unique())} unique")
    print(
        f"  - NAVD88 offset range: {df['mean_navd88_offset'].min():.3f} - {df['mean_navd88_offset'].max():.3f} m"
    )

    return df


def validate_u_and_v(ds):
    """
    Validate velocity components in an xarray Dataset to ensure they exist
    and have compatible units.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing 'u' (eastward) and 'v' (northward) velocity components

    Raises
    ------
    KeyError
        If velocity components are missing
    ValueError
        If units are undefined or incompatible
    """
    if not all(var in ds for var in ["u", "v"]):
        raise KeyError("Dataset must contain both 'u' and 'v' velocity components")

    # Verify units match
    u_units = ds.u.attrs.get("units")
    v_units = ds.v.attrs.get("units")

    if u_units is None:
        raise ValueError("Input dataset `u` units must be defined!")
    if v_units is None:
        raise ValueError("Input dataset `v` units must be defined!")

    if u_units and v_units and u_units != v_units:
        raise ValueError(f"Units mismatch: u: {u_units}, v: {v_units}")


def calculate_sea_water_speed(ds, config):
    """
    Calculate sea water speed from velocity components using vector magnitude.

    This function computes the magnitude of the horizontal velocity vector
    using the Pythagorean theorem: speed = √(u² + v²). The result represents
    the total horizontal speed regardless of direction.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing:
        - 'u': eastward sea water velocity component
        - 'v': northward sea water velocity component
        Velocities must have the same units (typically m/s)

    Returns
    -------
    xarray.Dataset
        Original dataset with added 'sea_water_speed' variable and CF-compliant metadata

    Raises
    ------
    KeyError
        If velocity components are missing
    ValueError
        If velocity component units are undefined or incompatible
    """
    validate_u_and_v(ds)

    output_variable_name = output_names["speed"]

    # Calculate speed maintaining original dimensions
    ds[output_variable_name] = np.sqrt(ds.u**2 + ds.v**2)

    specified_attrs = config["derived_vap_specification"]["speed"]["attributes"]

    # Add CF-compliant metadata
    ds[output_variable_name].attrs = {
        **specified_attrs,
        "additional_processing": (
            "Speed is calculated using the vector magnitude equation: speed = √(u² + v²), "
            "where u is eastward velocity and v is northward velocity."
        ),
        "computation": "sea_water_speed = np.sqrt(u**2 + v**2)",
        "input_variables": (
            "u: eastward_sea_water_velocity (m/s), "
            "v: northward_sea_water_velocity (m/s)"
        ),
    }

    return ds


def calculate_utc_timezone_offset(ds, config):
    """
    Populate UTC timezone offset for each face using precomputed values.

    This function reads the timezone offset for each face from precomputed data
    stored during the face center precalculation phase. The precomputed values
    are validated against the dataset's face indices and coordinates.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing face coordinate variables (used for validation)
    config : dict
        Configuration dictionary containing location information

    Returns
    -------
    xarray.Dataset
        Original dataset with added 'vap_utc_timezone_offset' variable and CF-compliant metadata

    Raises
    ------
    ValueError
        If precomputed data file doesn't exist or validation fails
    KeyError
        If required coordinate variables are missing
    """
    print("Loading precomputed timezone offset data...")

    # Get location key from config
    location_key = config.get("location_key") or config.get("location")
    if not location_key:
        raise ValueError("Config must contain 'location_key' or 'location' field")

    # Load precomputed face data with validation
    face_data_df = _load_precomputed_face_data(ds, config, location_key)

    # Extract timezone offset values
    timezone_offset_values = face_data_df["timezone_offset"].values.astype(np.int16)

    # Report statistics
    unique_offsets = np.unique(timezone_offset_values)
    print(f"Loaded timezone offsets for {len(timezone_offset_values)} faces")
    print(f"Found {len(unique_offsets)} unique timezone offsets: {unique_offsets}")

    output_variable_name = output_names["utc_timezone_offset"]

    # Create DataArray with timezone offset for each face
    ds[output_variable_name] = xr.DataArray(
        timezone_offset_values,
        dims=["face"],
        coords={
            "face": ds.face,
            "lat_center": ds.lat_center,
            "lon_center": ds.lon_center,
        },
    )

    # Add CF-compliant metadata
    ds[output_variable_name].attrs = {
        "long_name": "UTC Timezone Offset",
        "standard_name": "utc_offset",
        "units": "hours",
        "description": (
            "The offset in hours from Coordinated Universal Time (UTC) for the "
            "timezone at each face's geographic location. Each face receives its "
            "own timezone offset based on its specific lat_center/lon_center coordinates."
        ),
        "computation": "Loaded from precomputed face center data",
        "input_variables": "lat_center, lon_center coordinates for each face",
        "methodology": (
            "Timezone offset values were precomputed during face center calculation "
            "using TimezoneFinder library with each face's specific coordinates, "
            "then stored for efficient reuse."
        ),
        "unique_offsets_found": unique_offsets.tolist(),
        "coordinates": "face lat_center lon_center",
    }

    return ds


def calculate_distance_to_shore(ds, config):
    """
    Calculate distance to shore for each face using precomputed values.

    This function reads the distance to shore for each face from precomputed data
    stored during the face center precalculation phase. The precomputed values
    are validated against the dataset's face indices and coordinates.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing face coordinate variables (used for validation)
    config : dict
        Configuration dictionary containing location information

    Returns
    -------
    xarray.Dataset
        Original dataset with added 'vap_distance_to_shore' variable and CF-compliant metadata

    Raises
    ------
    ValueError
        If precomputed data file doesn't exist or validation fails
    KeyError
        If required coordinate variables are missing
    """
    print("Loading precomputed distance to shore data...")

    # Get location key from config
    location_key = config.get("location_key") or config.get("location")
    if not location_key:
        raise ValueError("Config must contain 'location_key' or 'location' field")

    # Load precomputed face data with validation
    face_data_df = _load_precomputed_face_data(ds, config, location_key)

    # Extract distance to shore values
    distance_values = face_data_df["distance_to_shore"].values.astype(np.float32)

    # Report statistics
    on_land_count = np.sum(distance_values == 0.0)
    max_distance_count = np.sum(distance_values == 200.0)
    print(f"Loaded distance to shore for {len(distance_values)} faces")
    print(f"  Points on land (distance=0): {on_land_count}")
    print(f"  Points at maximum distance (200NM): {max_distance_count}")
    print(
        f"  Distance range: {distance_values.min():.2f} - {distance_values.max():.2f} nautical miles"
    )

    output_variable_name = output_names["distance_to_shore"]

    # Create DataArray with distance to shore for each face
    ds[output_variable_name] = xr.DataArray(
        distance_values,
        dims=["face"],
        coords={
            "face": ds.face,
            "lat_center": ds.lat_center,
            "lon_center": ds.lon_center,
        },
    )

    # Add CF-compliant metadata
    ds[output_variable_name].attrs = {
        "long_name": "Distance to Shore",
        "standard_name": "distance_to_shore",
        "units": "nautical_miles",
        "description": (
            "The minimum distance from each face center to the nearest coastline. "
            "Calculated using high-resolution Natural Earth land polygon data. "
            "Points on land have distance=0."
        ),
        "computation": "Loaded from precomputed face center data",
        "input_variables": "lat_center, lon_center coordinates for each face",
        "methodology": (
            "Distance values were precomputed during face center calculation using "
            "World Equidistant Cylindrical (EPSG:4087) coordinate system for accurate "
            "distance calculation. Distance was calculated to the nearest land polygon "
            "using Shapely geometric operations with Natural Earth data, then converted "
            "from meters to nautical miles and stored for efficient reuse."
        ),
        "data_source": "Natural Earth 1:10m Land Polygons",
        "spatial_reference": "WGS84 (EPSG:4326) input, EPSG:4087 for distance calculation",
        "coordinates": "face lat_center lon_center",
        "points_on_land": int(on_land_count),
        "points_at_max_distance": int(max_distance_count),
    }

    return ds


def calculate_jurisdiction(ds, config):
    """
    Calculate maritime jurisdiction for each face using precomputed values.

    This function reads the maritime jurisdiction for each face from precomputed data
    stored during the face center precalculation phase. The precomputed values
    are validated against the dataset's face indices and coordinates.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing face coordinate variables (used for validation)
    config : dict
        Configuration dictionary containing location information

    Returns
    -------
    xarray.Dataset
        Original dataset with added variable and CF-compliant metadata:
        - 'vap_jurisdiction': Maritime jurisdiction classification

    Raises
    ------
    ValueError
        If precomputed data file doesn't exist or validation fails
    KeyError
        If required coordinate variables are missing
    """
    print("Loading precomputed maritime jurisdiction data...")

    # Get location key from config
    location_key = config.get("location_key") or config.get("location")
    if not location_key:
        raise ValueError("Config must contain 'location_key' or 'location' field")

    # Load precomputed face data with validation
    face_data_df = _load_precomputed_face_data(ds, config, location_key)

    # Extract jurisdiction values
    jurisdiction_values = face_data_df["jurisdiction"].values

    output_variable_name = output_names["jurisdiction"]

    # Get metadata from jurisdiction calculator
    jurisdiction_calculator = JurisdictionCalculator(config)

    # Create DataArray with jurisdiction for each face
    ds[output_variable_name] = xr.DataArray(
        jurisdiction_values,
        dims=["face"],
        coords={
            "face": ds.face,
            "lat_center": ds.lat_center,
            "lon_center": ds.lon_center,
        },
    )

    # Use metadata from jurisdiction calculator for CF-compliant attributes
    ds[output_variable_name].attrs = jurisdiction_calculator.get_metadata()

    return ds


def calculate_sea_water_to_direction(
    ds, config, direction_undefined_speed_threshold_ms=0.01
):
    """
    Calculate the direction sea water is flowing TO in compass convention.

    For a velocity vector with u=1, v=0 (flowing eastward):
    - to_direction = 90° (water flowing toward the east)

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset with 'u', 'v', and 'speed' variables
    config : dict
        Configuration dictionary
    direction_undefined_speed_threshold_ms : float, optional
        Speed threshold below which direction is set to NaN, default 0.01 m/s

    Returns
    -------
    xarray.Dataset
        Input dataset with added 'to_direction' variable
    """
    validate_u_and_v(ds)

    if output_names["speed"] not in ds.variables:
        raise KeyError(
            f"Dataset must contain '{output_names['speed']}'. "
            "Please run calculate_sea_water_speed() first."
        )

    # Calculate cartesian angle (counterclockwise from east)
    # - np.arctan2(y, x) takes arguments as (northward, eastward) = (v, u)
    # - Returns angle in radians with range [-π, π]
    # - Convert to degrees with range [-180°, 180°]
    cartesian_angle_degrees = np.rad2deg(np.arctan2(ds.v, ds.u))

    # Convert from cartesian angle to compass 'to' direction:
    # Example: u=1, v=0 (east) has cartesian angle 0°
    # 90 - 0 = 90° (pointing east)
    compass_to_direction_degrees = np.mod(90 - cartesian_angle_degrees, 360)

    compass_direction_degrees_expected_max = 360
    compass_direction_degrees_expected_min = 0

    compass_direction_degrees_max = np.max(compass_to_direction_degrees)
    compass_direction_degrees_min = np.min(compass_to_direction_degrees)

    if compass_direction_degrees_max > compass_direction_degrees_expected_max:
        raise ValueError(
            f"Maximum compass direction value {compass_direction_degrees_max}° "
            f"exceeds expected maximum of {compass_direction_degrees_expected_max}°"
        )

    if compass_direction_degrees_min < compass_direction_degrees_expected_min:
        raise ValueError(
            f"Minimum compass direction value {compass_direction_degrees_min}° "
            f"is below expected minimum of {compass_direction_degrees_expected_min}°"
        )

    # Set directions to NaN where speed is below threshold
    compass_to_direction_degrees = xr.where(
        ds[output_names["speed"]] > direction_undefined_speed_threshold_ms,
        compass_to_direction_degrees,
        np.nan,
    )

    output_variable_name = output_names["to_direction"]

    ds[output_variable_name] = compass_to_direction_degrees

    specified_attrs = config["derived_vap_specification"]["to_direction"]["attributes"]

    # Add CF-compliant metadata
    ds[output_variable_name].attrs = {
        **specified_attrs,
        "direction_reference": (
            "Reference table for velocity components and resulting to_direction:\n"
            "| Eastward (u) | Northward (v) | To Direction | Cardinal Direction |\n"
            "| 1            | 0             | 90           | East               |\n"
            "| 1            | 1             | 45           | Northeast          |\n"
            "| 0            | 1             | 0            | North              |\n"
            "| -1           | 1             | 315          | Northwest          |\n"
            "| -1           | 0             | 270          | West               |\n"
            "| -1           | -1            | 225          | Southwest          |\n"
            "| 0            | -1            | 180          | South              |\n"
            "| 1            | -1            | 135          | Southeast          |\n"
            "| 0            | 0             | undefined    | undefined          |"
        ),
        "additional_processing": (
            f"Directions set to NaN for speeds below {direction_undefined_speed_threshold_ms} m/s."
        ),
        "computation": (
            "cartesian_angle_degrees = np.rad2deg(np.arctan2(ds.v, ds.u))\n"
            "compass_to_direction_degrees = np.mod(90 - cartesian_angle_degrees, 360)\n"
        ),
        "input_variables": (
            "u: eastward_sea_water_velocity (m/s), "
            "v: northward_sea_water_velocity (m/s)"
        ),
    }

    return ds


def calculate_sea_water_power_density(ds, config, rho: float = 1025.0):
    """
    Calculate sea water power density from velocity components.

    This function computes the power per unit area available in the flow using the
    fundamental fluid dynamics equation P = ½ρv³.
    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing 'sea_water_speed' (must be previously calculated from
        FVCOM u,v velocity components)
    rho : float, optional
        Water density in kg/m³. Defaults to 1025.0 kg/m³, which is typical for
        seawater at standard temperature and pressure.

    Returns
    -------
    xarray.Dataset
        Original dataset with added 'sea_water_power_density' variable

    Raises
    ------
    KeyError
        If 'sea_water_speed' is not present in dataset
    """

    if output_names["speed"] not in ds.variables:
        raise KeyError(
            f"Dataset must contain '{output_names['speed']}'. "
            "Please run calculate_sea_water_speed() first."
        )

    output_variable_name = output_names["power_density"]

    # Calculate power density using Equation 1 from
    # Haas, Kevin A., et al. "Assessment of Energy Production Potential from
    # Tidal Streams in the United States." , Jun. 2011. https://doi.org/10.2172/1219367
    ds[output_variable_name] = 0.5 * rho * ds[output_names["speed"]] ** 3

    specified_attrs = config["derived_vap_specification"]["power_density"]["attributes"]

    # Add CF-compliant metadata
    ds[output_variable_name].attrs = {
        **specified_attrs,
        "additional_processing": (
            "Computed using the fluid power density equation P = ½ρv³ with seawater "
            f"density ρ = {rho} kg/m³. The calculation uses sea water speed derived "
            "from FVCOM u,v velocity components written to the sea_water_speed variable."
        ),
        "computation": "sea_water_power_density = 0.5 * rho * sea_water_speed**3",
        "input_variables": (f"sea_water_speed (m/s), rho=`{rho}` (kg/m³)"),
        "citation": (
            "Haas, Kevin A., et al. 'Assessment of Energy Production Potential "
            "from Tidal Streams in the United States.' Georgia Tech Research "
            "Corporation, Jun. 2011. https://doi.org/10.2172/1219367"
        ),
    }

    return ds


def calculate_zeta_center(ds):
    """
    Calculate sea surface elevation at cell centers from node values.
    """

    # Convert to zero-based indexing
    nv_values = ds.nv.values - 1

    # Get raw zeta values (all timesteps)
    zeta_values = ds.zeta.values  # shape (n_times, n_nodes)

    # Create empty result array
    n_times = zeta_values.shape[0]
    n_faces = nv_values.shape[1]
    result_array = np.zeros((n_times, n_faces), dtype=ds.zeta.dtype)

    # For each of the 3 nodes of each face
    for i in range(3):
        # Get indices for the i-th node of all faces
        node_indices = nv_values[i, :]  # shape (n_faces,)

        # For all timesteps, add zeta values at these nodes to result
        # This vectorized operation processes all timesteps at once
        result_array += zeta_values[:, node_indices]

    # Divide by 3 to get the average
    result_array /= 3.0

    # Create DataArray with the results
    zeta_center = xr.DataArray(
        result_array,
        dims=("time", "face"),
        coords={
            "time": ds.time,
            "lon_center": ds.lon_center,
            "lat_center": ds.lat_center,
        },
    )

    # Add attributes
    zeta_center.attrs = {
        **ds.zeta.attrs,
        "long_name": "Sea Surface Height at Cell Centers from NAVD88",
        "standard_name": "sea_surface_elevation_above_geoid",
        "coordinates": "time face",
        "mesh": "cell_centered",
        "units": "m from NAVD88",
        "interpolation": (
            "Computed by averaging the surface elevation values from the three "
            "nodes that define each triangular cell using the node-to-cell "
            "connectivity array (nv)"
        ),
        "computation": "zeta_center = mean(zeta[nv - 1], axis=1)",
        "input_variables": "zeta: sea_surface_height_above_geoid at nodes",
    }

    ds[output_names["zeta_center"]] = zeta_center

    return ds


def calculate_surface_elevation_and_depths(ds, offset_file_path):
    """
    Calculate surface elevation and depths using pre-computed NAVD88 offset.

    This function loads the pre-computed mean NAVD88 offset and:
    1. Calculates surface elevation relative to mean sea level
    2. Converts bathymetry to MSL reference frame
    3. Calculates sigma-level depths and seafloor depth using consistent MSL reference

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing:
        - zeta_center (must be calculated first)
        - h_center: bathymetry at cell centers
        - sigma_layer: sigma level coordinates (for depth calculation)
    offset_file_path : str or Path
        Path to the stored mean NAVD88 offset NetCDF file

    Returns
    -------
    xarray.Dataset
        Dataset with added variables:
        - 'surface_elevation': surface height relative to MSL
        - 'depth': sigma-level depths (if sigma_layer present)
        - 'seafloor_depth': total water column depth

    Raises
    ------
    ValueError
        If required variables not found or coordinates mismatch
    FileNotFoundError
        If offset file doesn't exist
    """

    # Check required variables
    if output_names["zeta_center"] not in ds.variables:
        raise ValueError(
            f"Dataset must contain '{output_names['zeta_center']}' variable"
        )

    if output_names["h_center"] not in ds.variables:
        raise ValueError(f"Dataset must contain '{output_names['h_center']}' variable")

    # Load the stored offset
    if not offset_file_path.exists():
        raise FileNotFoundError(f"Offset file not found: {offset_file_path}")

    print(f"Loading mean NAVD88 offset from: {offset_file_path}")
    with xr.open_dataset(offset_file_path) as offset_ds:
        mean_offset = offset_ds.mean_navd88_offset

        # Validate coordinate consistency
        zeta_center = ds[output_names["zeta_center"]]

        if len(mean_offset.face) != len(zeta_center.face):
            raise ValueError(
                f"Face dimension mismatch: offset file has {len(mean_offset.face)} faces, "
                f"but dataset has {len(zeta_center.face)} faces"
            )

        # Check coordinate consistency
        if not np.allclose(
            mean_offset.lon_center.values, zeta_center.lon_center.values, rtol=1e-10
        ):
            raise ValueError(
                "Longitude coordinates mismatch between dataset and offset file"
            )

        if not np.allclose(
            mean_offset.lat_center.values, zeta_center.lat_center.values, rtol=1e-10
        ):
            raise ValueError(
                "Latitude coordinates mismatch between dataset and offset file"
            )

        # 1. Calculate surface elevation relative to MSL
        surface_elevation_values = zeta_center - mean_offset

        surface_elevation = xr.DataArray(
            surface_elevation_values.values,
            dims=("time", "face"),
            coords={
                "time": ds.time,
                "lon_center": ds.lon_center,
                "lat_center": ds.lat_center,
                "face": zeta_center.face,
            },
        )

        surface_elevation.attrs = {
            "long_name": "Sea Surface Elevation Relative to Mean Sea Level",
            "name": "sea_surface_height_above_mean_sea_level",
            "units": "m",
            "coordinates": "time face lon_center lat_center",
            "mesh": "cell_centered",
            "description": (
                "Sea surface elevation calculated by subtracting the temporal mean "
                "NAVD88 offset from instantaneous zeta_center values. This represents "
                "surface elevation fluctuations relative to mean sea level and should "
                "have a temporal mean near zero (when using complete yearly data)."
            ),
            "computation": "surface_elevation = zeta_center - mean_navd88_offset",
            "offset_source": str(offset_file_path),
            "reference_level": "mean_sea_level",
            "input_variables": (
                f"zeta_center: {output_names['zeta_center']}, "
                "mean_navd88_offset: pre-computed temporal mean"
            ),
        }

        ds[output_names["surface_elevation"]] = surface_elevation

        # 2. Calculate depth from sea surface to seafloor
        # This is the actual depth measured from the instantaneous water surface
        depth_from_sea_surface = ds[output_names["h_center"]] + zeta_center

        # Ensure proper dimension ordering
        depth_from_sea_surface = depth_from_sea_surface.transpose("time", "face")
        depth_from_sea_surface.attrs = {
            "long_name": "Water Depth from Sea Surface to Seafloor",
            "standard_name": "sea_floor_depth_below_sea_surface",
            "units": ds[output_names["h_center"]].attrs["units"],
            "positive": "down",
            "coordinates": "time face lon_center lat_center",
            "mesh": "cell_centered",
            "description": (
                "The water depth representing the actual distance from the "
                "instantaneous water surface to the seafloor. This is the depth that "
                "would be measured by instruments from the current sea surface. "
                "The depth varies with tidal conditions as the water surface elevation changes."
            ),
            "computation": "depth_from_sea_surface = h_center + zeta_center",
            "reference_level": "instantaneous_sea_surface",
            "input_variables": (
                "h_center: sea_floor_depth_below_geoid (m), "
                "zeta_center: sea_surface_height_above_geoid (m)"
            ),
            "application": "depth_measurements_from_surface",
            "note": (
                "Both h_center and zeta_center use the same geoid reference (NAVD88), "
                "so they can be added directly to get depth from surface without datum conversions."
            ),
        }

        ds[output_names["sea_floor_depth"]] = depth_from_sea_surface

        # 4. Calculate sigma-level depths if sigma coordinates are available
        print("Calculating sigma-level depths...")

        # Extract sigma levels
        sigma_layer = ds.sigma_layer.T.values[0]
        sigma_3d = sigma_layer.reshape(1, -1, 1)

        # Expand depth from sea surface for sigma calculation
        depth_3d = depth_from_sea_surface.expand_dims(
            dim={"sigma_layer": len(sigma_layer)}, axis=1
        )

        # Calculate depth at each sigma level (positive down from surface)
        sigma_depths = -(depth_3d * sigma_3d)

        sigma_depths.attrs = {
            "long_name": "Depth Below Sea Surface at Sigma Levels",
            "standard_name": "depth",
            "units": ds[output_names["h_center"]].attrs["units"],
            "positive": "down",
            "coordinates": "time sigma_layer face lon_center lat_center",
            "mesh": "cell_centered",
            "description": (
                "Depth at each sigma level calculated using water column depth from sea surface. "
                "This represents the actual physical depth below the instantaneous water "
                "surface that would be measured by instruments at each sigma level. "
                "Depths vary with tidal conditions as the total water column depth changes."
            ),
            "computation": "depth = -(h_center + zeta_center) * sigma_coordinate",
            "reference_level": "instantaneous_sea_surface",
            "input_variables": (
                "h_center: sea_floor_depth_below_geoid (m), "
                "zeta_center: sea_surface_height_above_geoid (m), "
                "sigma: ocean_sigma_coordinate"
            ),
            "application": "depth_measurements_from_surface",
            "sigma_convention": "sigma=0 at surface, sigma=-1 at bottom",
        }

        ds[output_names["depth"]] = sigma_depths
        print("Sigma-level depths calculated successfully using depth from sea surface")

        print("Surface elevation and depth from sea surface calculated successfully")
        print(
            f"Surface elevation statistics: "
            f"min={surface_elevation.min().values:.3f}m, "
            f"max={surface_elevation.max().values:.3f}m, "
            f"temporal_mean={surface_elevation.mean().values:.6f}m"
        )
        print(
            f"Depth from sea surface statistics: "
            f"min={depth_from_sea_surface.min().values:.3f}m, "
            f"max={depth_from_sea_surface.max().values:.3f}m"
        )

    return ds


def calculate_top_face_area_of_fvcom_volume_from_coordinates(ds):
    """
    Calculate the top face area of the of the FVCOM triangular prism volumes

    The top face area is also the bottom face area

    Parameters
    ----------
    ds : xarray.Dataset
        FVCOM dataset containing mesh information

    Returns
    -------
    numpy.ndarray
        Array of element areas in square meters
    """
    # Get the node indices for each face
    nv = ds.nv.values - 1  # Convert to 0-based indexing

    # Get node coordinates
    lon_node = ds.lon_node.values
    lat_node = ds.lat_node.values

    # Constants for Earth calculations
    R_EARTH = 6371000  # Earth radius in meters
    DEG_TO_RAD = np.pi / 180

    # Convert all nodes to radians
    lat_rad = lat_node * DEG_TO_RAD
    lon_rad = lon_node * DEG_TO_RAD

    # Convert all nodes to cartesian coordinates
    x = R_EARTH * np.cos(lat_rad) * np.cos(lon_rad)
    y = R_EARTH * np.cos(lat_rad) * np.sin(lon_rad)
    z = R_EARTH * np.sin(lat_rad)

    # Get coordinates for each node of each triangle
    # Using advanced indexing to extract the coordinates for all triangles at once
    x1 = x[nv[0, :]]
    y1 = y[nv[0, :]]
    z1 = z[nv[0, :]]

    x2 = x[nv[1, :]]
    y2 = y[nv[1, :]]
    z2 = z[nv[1, :]]

    x3 = x[nv[2, :]]
    y3 = y[nv[2, :]]
    z3 = z[nv[2, :]]

    # Calculate vectors for sides of triangles (vectorized)
    # First vector: from point 1 to point 2
    v1x = x2 - x1
    v1y = y2 - y1
    v1z = z2 - z1

    # Second vector: from point 1 to point 3
    v2x = x3 - x1
    v2y = y3 - y1
    v2z = z3 - z1

    # Cross product components (vectorized)
    crossx = v1y * v2z - v1z * v2y
    crossy = v1z * v2x - v1x * v2z
    crossz = v1x * v2y - v1y * v2x

    # Calculate area using magnitude of cross product
    top_face_areas = 0.5 * np.sqrt(crossx**2 + crossy**2 + crossz**2)

    return top_face_areas


def calculate_element_volume(ds):
    """
    Calculate volumes for each element at each time step and sigma layer

    Parameters
    ----------
    ds : xarray.Dataset
        FVCOM dataset containing bathymetry, surface elevation, and mesh information

    Returns
    -------
    xarray.Dataset
        Original dataset with added 'element_volume' variable
    """
    # Calculate element areas
    top_face_areas = calculate_top_face_area_of_fvcom_volume_from_coordinates(ds)

    # Get dimensions
    n_time = len(ds.time)
    n_sigma_layer = len(ds.sigma_layer)
    n_face = len(ds.face)

    # Get bathymetry and sea surface height
    h_center = ds[
        output_names["sea_floor_depth"]
    ].values  # Bathymetry at element centers
    zeta_center = ds[
        output_names["surface_elevation"]
    ].values  # Surface elevation at element centers

    # Get node indices for each element
    nv = ds.nv.values - 1  # Convert to 0-based indexing

    # Get sigma layer and level values
    sigma_layer_values = ds.sigma_layer.values  # Shape: (n_sigma_layer, n_node)
    sigma_level_values = ds.sigma_level.values  # Shape: (n_sigma_level, n_node)

    # Calculate layer thicknesses at nodes (vectorized)
    # Layer thickness is the difference between adjacent sigma levels
    layer_thickness_at_nodes = np.abs(
        sigma_level_values[1 : n_sigma_layer + 1, :]
        - sigma_level_values[:n_sigma_layer, :]
    )

    # Create indices arrays for advanced indexing
    # We need to get the thickness at each of the three nodes for each face, for each layer

    # Create a 3D array where each slice represents one layer
    # and contains the node indices for all faces
    layer_indices = np.zeros((n_sigma_layer, 3, n_face), dtype=int)
    for i in range(3):
        layer_indices[:, i, :] = nv[i, :]

    # Create a meshgrid for the sigma layers
    layer_mesh = np.arange(n_sigma_layer)[:, np.newaxis, np.newaxis]
    layer_mesh = np.broadcast_to(layer_mesh, (n_sigma_layer, 3, n_face))

    # Get all layer thicknesses for all nodes of all elements in one operation
    # This will be shape (n_sigma_layer, 3, n_face)
    all_node_thicknesses = layer_thickness_at_nodes[layer_mesh, layer_indices]

    # Average the thickness of the three nodes for each element and each layer
    # Result shape: (n_sigma_layer, n_face)
    element_thickness_fraction = np.mean(all_node_thicknesses, axis=1)

    # Calculate total water depth at each time step
    # Shape: (n_time, n_face)
    total_depth = np.abs(h_center) + zeta_center

    # Reshape arrays for broadcasting
    # element_areas: (n_face) -> (1, n_sigma_layer, n_face)
    element_areas_broadcast = top_face_areas.reshape(1, 1, n_face).repeat(
        n_sigma_layer, axis=1
    )

    # total_depth: (n_time, n_face) -> (n_time, 1, n_face)
    total_depth_broadcast = total_depth.reshape(n_time, 1, n_face)

    # element_thickness_fraction: (n_sigma_layer, n_face) -> (1, n_sigma_layer, n_face)
    element_thickness_fraction_broadcast = element_thickness_fraction.reshape(
        1, n_sigma_layer, n_face
    )

    # Calculate all volumes in a single vectorized operation
    # element_volumes shape: (n_time, n_sigma_layer, n_face)
    element_volumes = (
        element_areas_broadcast
        * total_depth_broadcast
        * element_thickness_fraction_broadcast
    )

    # Add element volumes to dataset
    ds[output_names["element_volume"]] = xr.DataArray(
        element_volumes,
        dims=["time", "sigma_layer", "face"],
        attrs={
            "long_name": "Element Volume",
            "standard_name": "volume_of_water_per_element",
            "units": "m^3",
            "description": "Volume of each triangular element at each time step and sigma layer",
            "methodology": "Calculated as element area multiplied by layer thickness using fully vectorized operations",
            "computation": "element_volume = element_area * total_water_depth * sigma_layer_thickness",
            "input_variables": "h_center: bathymetry (m), zeta_center: surface elevation (m), sigma_layer: sigma coordinate",
        },
    )

    return ds


def validate_depth_inputs(ds):
    """
    Validate required variables for depth calculations in an xarray Dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing depth-related variables

    Raises
    ------
    KeyError
        If required variables are missing
    ValueError
        If units or attributes are undefined or incompatible
    """
    h_center = output_names["h_center"]
    zeta_center = output_names["zeta_center"]

    required_vars = [h_center, zeta_center]

    if not all(var in ds for var in required_vars):
        raise KeyError(
            f"Dataset must contain both '{h_center}' (bathymetry) and '{zeta_center}' (surface elevation)"
        )

    # Verify h_center attributes
    if "units" not in ds[h_center].attrs:
        raise ValueError("Input dataset `h_center` units must be defined!")
    if "positive" not in ds[h_center].attrs:
        raise ValueError("Input dataset `h_center` must define positive direction!")
    if ds[h_center].attrs["positive"] != "down":
        raise ValueError("Input dataset `h_center` positive direction must be 'down'!")

    # Verify zeta_center attributes
    if "units" not in ds[zeta_center].attrs:
        raise ValueError("Input dataset `zeta_center` units must be defined!")
    if "positive" not in ds[zeta_center].attrs:
        raise ValueError("Input dataset `zeta_center` must define positive direction!")
    if ds[zeta_center].attrs["positive"] != "up":
        raise ValueError("Input dataset `zeta_center` positive direction must be 'up'!")

    # Verify units match
    h_units = ds[h_center].attrs["units"]
    z_units = ds[zeta_center].attrs["units"]
    if h_units != z_units:
        raise ValueError(f"Units mismatch: h_center: {h_units}, zeta_center: {z_units}")


def calculate_depth(ds):
    """
    Calculate depth at sigma levels for FVCOM output.

    This function computes the depth at each sigma level using the equation:
    depth = -(h + ζ) * σ

    where:
        h: bathymetry (positive down from geoid)
        ζ: sea surface elevation (positive up from geoid)
        σ: sigma level coordinates (0 at surface to -1 at bottom)

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing:
        - 'h_center': bathymetry at cell centers
        - 'zeta_center': water surface elevation at cell centers
        - 'sigma_layer': sigma level coordinates

    Returns
    -------
    xarray.Dataset
        Original dataset with added 'depth' variable and CF-compliant metadata

    Raises
    ------
    KeyError
        If required variables are missing
    ValueError
        If required attributes are missing or inconsistent
    """
    validate_depth_inputs(ds)

    if "sigma_layer" not in ds:
        raise KeyError("Dataset must contain 'sigma_layer' coordinates")

    # Extract sigma levels
    sigma_layer = ds.sigma_layer.T.values[0]

    # Reshape sigma to (1, sigma_layer, 1) for proper broadcasting
    sigma_3d = sigma_layer.reshape(1, -1, 1)

    # Calculate total water depth (h + zeta) and reshape
    total_depth = (
        ds[output_names["h_center"]] + ds[output_names["zeta_center"]]
    )  # Shape (time, face)
    total_depth_3d = total_depth.expand_dims(
        dim={"sigma_layer": len(sigma_layer)}, axis=1
    )

    # Calculate depth - this gives depths that are positive down from the surface
    ds[output_names["depth"]] = -(total_depth_3d * sigma_3d)

    # Add CF-compliant metadata
    ds[output_names["depth"]].attrs = {
        "long_name": "Depth Below Sea Surface",
        "standard_name": "depth",
        "units": ds[output_names["h_center"]].attrs["units"],
        "positive": "down",
        "coordinates": "time cell sigma",
        "description": (
            "Depth represents the vertical distance below the sea surface at each sigma "
            "level, varying with both the fixed bathymetry and time-varying surface elevation."
        ),
        "methodology": (
            "Depth is calculated using the sigma coordinate transformation: "
            "depth = -(h + ζ) * σ, where h is bathymetry, ζ is surface elevation, "
            "and σ is the sigma level coordinate. This makes depth=0 at the surface (σ=0) "
            "and depth=h+ζ at the bottom (σ=-1)."
        ),
        "computation": "depth = -(h_center + zeta_center) * sigma",
        "input_variables": (
            "h_center: sea_floor_depth_below_geoid (m), "
            "zeta_center: sea_surface_height_above_geoid (m), "
            "sigma: ocean_sigma_coordinate"
        ),
    }

    return ds


def calculate_sea_floor_depth(ds):
    """
    Calculate sea floor depth below sea surface from FVCOM output.

    This function computes the instantaneous water column depth - the vertical
    distance between the sea surface and the seabed at each time step. The result
    accounts for both the fixed bathymetry and time-varying sea surface height.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing:
        - 'h_center': bathymetry at cell centers (positive down from geoid)
        - 'zeta_center': water surface elevation at cell centers (positive up from geoid)

    Returns
    -------
    xarray.Dataset
        Original dataset with added 'seafloor_depth' variable and CF-compliant metadata

    Raises
    ------
    KeyError
        If required variables are missing
    ValueError
        If required attributes are missing or inconsistent
    """
    validate_depth_inputs(ds)

    # Calculate total water column depth
    # Since h_center is already positive down from geoid and zeta_center is positive up,
    # the correct formula is h_center + zeta_center to get the total water depth
    ds[output_names["sea_floor_depth"]] = (
        ds[output_names["h_center"]] + ds[output_names["zeta_center"]]
    )

    # Add CF-compliant metadata
    ds[output_names["sea_floor_depth"]].attrs = {
        "long_name": "Sea Floor Depth Below Sea Surface",
        "standard_name": "sea_floor_depth_below_sea_surface",
        "units": ds[output_names["h_center"]].attrs["units"],
        "positive": "down",
        "description": (
            "The vertical distance between the sea surface and the seabed as measured "
            "at a given point in space including the variance caused by tides."
        ),
        "methodology": (
            "Total water column depth is calculated by combining the fixed bathymetry "
            "and time-varying surface elevation. Since h_center is positive down "
            "from geoid and zeta_center is positive up from geoid, "
            "we add them to get the total water column depth."
        ),
        "computation": "seafloor_depth = h_center + zeta_center",
        "input_variables": (
            "h_center: sea_floor_depth_below_geoid (m), "
            "zeta_center: sea_surface_height_above_geoid (m)"
        ),
    }

    return ds


def calculate_depth_average(ds, variable_name):
    """
    Calculate depth average for a given variable

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing the variable to be averaged
    variable_name : str
        Name of variable to average across depth

    Returns
    -------
    xarray.Dataset
        Dataset with added depth-averaged variable
    """
    this_output_name = output_names[variable_name]
    sanitized_this_output_name = this_output_name.replace("vap_", "")

    if this_output_name not in ds:
        raise KeyError(f"Dataset must contain '{this_output_name}'")

    # Calculate depth average
    # depth_avg_name = f"{variable_name}_depth_avg"
    depth_avg_name = f"{output_names['mean']}_{sanitized_this_output_name}"

    ds[depth_avg_name] = ds[this_output_name].mean(dim="sigma_layer")

    # Copy and modify attributes for averaged variable
    # Start with original attributes but remove standard_name if it exists
    attrs = ds[this_output_name].attrs.copy()
    attrs.pop("standard_name", None)

    ds[depth_avg_name].attrs = {
        **attrs,
        "long_name": f"Water column mean {ds[this_output_name].attrs.get('long_name', this_output_name)}",
        "statistical_computation": "Mean across sigma layers",
    }

    return ds


def calculate_depth_statistics(
    ds, variable_name, stats_to_calculate=None, face_chunk_size=7500
):
    """
    Calculate depth statistics for a given variable.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing the variable to be analyzed
    variable_name : str
        Name of variable to calculate statistics for
    stats_to_calculate : list, optional
        List of statistics to calculate. Options: 'mean', 'median', 'max', 'p95'
        If None, all statistics will be calculated
    face_chunk_size : int, optional
        Number of faces to process at once (default: 10000)
        Adjust based on available memory - smaller values use less memory but may be slower

    Returns
    -------
    xarray.Dataset
        Dataset with added depth statistics variables
    """
    this_output_name = output_names[variable_name]
    sanitized_this_output_name = this_output_name.replace("vap_", "")

    if this_output_name not in ds:
        raise KeyError(f"Dataset must contain '{this_output_name}'")

    valid_stats = ["mean", "median", "max", "p95"]

    # Default to calculating all statistics if not specified
    if stats_to_calculate is None:
        stats_to_calculate = ["mean", "max", "p95"]

    # Validate stats_to_calculate
    for stat in stats_to_calculate:
        if stat not in valid_stats:
            raise ValueError(
                f"Invalid statistic: {stat}. Must be one of: {valid_stats}"
            )

    dim = "sigma_layer"
    percentile_value = 95

    # Define output variable names
    depth_avg_name = f"{output_names['mean']}_{sanitized_this_output_name}"
    depth_median_name = f"{output_names['median']}_{sanitized_this_output_name}"
    depth_max_name = f"{output_names['max']}_{sanitized_this_output_name}"
    depth_percentile_name = f"{output_names['p95'].replace('<PERCENTILE>', str(percentile_value))}_{sanitized_this_output_name}"

    # Get original variable attributes
    orig_attrs = ds[this_output_name].attrs.copy()
    orig_long_name = orig_attrs.get("long_name", this_output_name)

    # Remove standard_name from attributes for all derived variables
    clean_attrs = orig_attrs.copy()
    clean_attrs.pop("standard_name", None)

    # Get dimensions info
    time_dim = ds.sizes["time"]
    face_dim = ds.sizes["face"]

    # Get dimensions without depth for creating result arrays
    dims_without_depth = [d for d in ds[this_output_name].dims if d != dim]

    # Initialize arrays for the requested statistics
    result_arrays = {}
    if "mean" in stats_to_calculate:
        result_arrays["mean"] = np.zeros((time_dim, face_dim), dtype=np.float32)
    if "median" in stats_to_calculate:
        result_arrays["median"] = np.zeros((time_dim, face_dim), dtype=np.float32)
    if "max" in stats_to_calculate:
        result_arrays["max"] = np.zeros((time_dim, face_dim), dtype=np.float32)
    if "p95" in stats_to_calculate:
        result_arrays["p95"] = np.zeros((time_dim, face_dim), dtype=np.float32)

    print(
        f"\t\tCalculating requested depth statistics: {', '.join(stats_to_calculate)}..."
    )

    # Single loop to process all statistics in chunks
    for face_start in range(0, face_dim, face_chunk_size):
        # Define current chunk range
        face_end = min(face_start + face_chunk_size, face_dim)

        # Extract chunk data for the current face slice
        chunk = ds[this_output_name].isel(face=slice(face_start, face_end))

        # Calculate each requested statistic for this chunk
        if "mean" in stats_to_calculate:
            mean_chunk = chunk.mean(dim=dim).values
            result_arrays["mean"][:, face_start:face_end] = mean_chunk

        if "median" in stats_to_calculate:
            median_chunk = chunk.median(dim=dim).values
            result_arrays["median"][:, face_start:face_end] = median_chunk

        # For max and percentile, we need the raw data array
        if "max" in stats_to_calculate or "p95" in stats_to_calculate:
            chunk_data = chunk.values
            sigma_axis = chunk.dims.index(dim)

            if "max" in stats_to_calculate:
                max_chunk = np.max(chunk_data, axis=sigma_axis)
                result_arrays["max"][:, face_start:face_end] = max_chunk

            if "p95" in stats_to_calculate:
                percentile_chunk = np.percentile(
                    chunk_data, percentile_value, axis=sigma_axis
                )
                result_arrays["p95"][:, face_start:face_end] = percentile_chunk

            # Explicitly release memory
            chunk_data = None

    # Create the DataArrays and set attributes for each statistic
    if "mean" in stats_to_calculate:
        print(f"\t\tAdding {depth_avg_name}...")
        ds[depth_avg_name] = (dims_without_depth, result_arrays["mean"])
        ds[depth_avg_name].attrs = {
            **clean_attrs,
            "long_name": f"Depth averaged {orig_long_name}",
            "statistical_computation": "Mean across sigma layers",
        }

    if "median" in stats_to_calculate:
        print(f"\t\tAdding {depth_median_name}...")
        ds[depth_median_name] = (dims_without_depth, result_arrays["median"])
        ds[depth_median_name].attrs = {
            **clean_attrs,
            "long_name": f"Depth median {orig_long_name}",
            "statistical_computation": "Median across sigma layers",
        }

    if "max" in stats_to_calculate:
        print(f"\t\tAdding {depth_max_name}...")
        ds[depth_max_name] = (dims_without_depth, result_arrays["max"])
        ds[depth_max_name].attrs = {
            **clean_attrs,
            "long_name": f"Depth maximum {orig_long_name}",
            "statistical_computation": "Maximum value across sigma layers",
        }

    if "p95" in stats_to_calculate:
        print(f"\t\tAdding {depth_percentile_name}...")
        ds[depth_percentile_name] = (dims_without_depth, result_arrays["p95"])
        ds[depth_percentile_name].attrs = {
            **clean_attrs,
            "long_name": f"Depth {percentile_value}th percentile {orig_long_name}",
            "statistical_computation": f"{percentile_value}th percentile across sigma layers",
        }

    return ds


def process_single_file(
    nc_file,
    config,
    location,
    surface_elevation_offset_path,
    output_dir,
    file_index,
    total_files=None,
    start_time=None,
):
    """Process a single netCDF file and save the results."""

    file_start_time = time.time()
    print(f"Calculating vap for {nc_file}")

    # Use context manager to ensure dataset is properly closed
    with xr.open_dataset(
        nc_file, engine=config["dataset"]["xarray_netcdf4_engine"]
    ) as this_ds:
        print(f"\t[{file_index}] Calculating speed...")
        this_ds = calculate_sea_water_speed(this_ds, config)

        print(f"\t[{file_index}] Calculating to direction...")
        this_ds = calculate_sea_water_to_direction(this_ds, config)

        print(f"\t[{file_index}] Calculating power density...")
        this_ds = calculate_sea_water_power_density(this_ds, config)

        print(f"\t[{file_index}] Calculating zeta_center...")
        this_ds = calculate_zeta_center(this_ds)

        print(f"\t[{file_index}] Calculating surface_elevation...")
        this_ds = calculate_surface_elevation_and_depths(
            this_ds, surface_elevation_offset_path
        )

        print(f"\t[{file_index}] Calculating timezone...")
        this_ds = calculate_utc_timezone_offset(this_ds, config)

        print(f"\t[{file_index}] Calculating distance to shore...")
        this_ds = calculate_distance_to_shore(this_ds, config)

        print(f"\t[{file_index}] Calculating jurisdiction...")
        this_ds = calculate_jurisdiction(this_ds, config)

        # print(f"\t[{file_index}] Calculating depth...")
        # this_ds = calculate_depth(this_ds)
        #
        # print(f"\t[{file_index}] Calculating sea_floor_depth...")
        # this_ds = calculate_sea_floor_depth(this_ds)

        print(f"\t[{file_index}] Calculating u water column average")
        this_ds = calculate_depth_statistics(this_ds, "u", stats_to_calculate=["mean"])

        print(f"\t[{file_index}] Calculating v water column average")
        this_ds = calculate_depth_statistics(this_ds, "v", stats_to_calculate=["mean"])

        print(f"\t[{file_index}] Calculating to_direction water column average")
        this_ds = calculate_depth_statistics(
            this_ds, "to_direction", stats_to_calculate=["mean"]
        )

        print(f"\t[{file_index}] Calculating speed depth average statistics")
        this_ds = calculate_depth_statistics(
            this_ds, "speed", stats_to_calculate=["mean", "max"]
        )

        print(f"\t[{file_index}] Calculating power_density depth average statistics")
        this_ds = calculate_depth_statistics(
            this_ds, "power_density", stats_to_calculate=["mean", "max"]
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
                this_ds,
                location["output_name"],
                config["dataset"]["name"],
                "b1",
                temporal=temporal_string,
            )
        )

        this_ds = attrs_manager.standardize_dataset_global_attrs(
            this_ds,
            config,
            location,
            "b1",
            [str(nc_file)],
        )

        output_path = Path(
            output_dir,
            # f"{file_index:03d}.{data_level_file_name}",
            data_level_file_name,
        )

        print(f"\t[{file_index}] Saving final results to {output_path}...")

        nc_manager.nc_write(this_ds, output_path, config)

        # Calculate time elapsed for this file
        file_elapsed_time = time.time() - file_start_time
        elapsed_hours, remainder = divmod(file_elapsed_time, 3600)
        elapsed_minutes, elapsed_seconds = divmod(remainder, 60)

        print(
            f"\t[{file_index}] File processed in {int(elapsed_hours):02d}:{int(elapsed_minutes):02d}:{int(elapsed_seconds):02d}"
        )

        # Calculate and display estimated time to completion if we have the total files info
        if (
            total_files is not None
            and start_time is not None
            and file_index < total_files
        ):
            files_completed = file_index
            files_remaining = total_files - files_completed

            # Calculate average time per file based on completed files
            total_elapsed_time = time.time() - start_time
            avg_time_per_file = total_elapsed_time / files_completed

            # Estimate time remaining
            est_time_remaining = avg_time_per_file * files_remaining
            est_hours, remainder = divmod(est_time_remaining, 3600)
            est_minutes, est_seconds = divmod(remainder, 60)

            # Calculate estimated completion time
            est_completion_time = datetime.now() + timedelta(seconds=est_time_remaining)

            print(
                f"\t[{file_index}] Progress: {files_completed}/{total_files} files ({files_completed / total_files * 100:.1f}%)"
            )
            print(
                f"\t[{file_index}] Average time per file: {int(avg_time_per_file // 60):02d}:{int(avg_time_per_file % 60):02d} (mm:ss)"
            )
            print(
                f"\t[{file_index}] Estimated time remaining: {int(est_hours):02d}:{int(est_minutes):02d}:{int(est_seconds):02d} (hh:mm:ss)"
            )
            print(
                f"\t[{file_index}] Estimated completion time: {est_completion_time.strftime('%Y-%m-%d %H:%M:%S')}"
            )

        return file_index


def get_processed_file_indices(output_dir):
    """
    Extract file indices from existing output files.
    Assumes output files are named like: "123.nc", "456.nc", etc.
    """
    processed_indices = set()
    existing_files = list(output_dir.rglob("*.nc"))

    for file_path in existing_files:
        filename = file_path.name
        # Extract the number before the first dot
        base_name = filename.split(".")[0]
        if base_name.isdigit():
            processed_indices.add(int(base_name))

    return processed_indices


def derive_vap(
    config,
    location_key,
    surface_elevation_offset_path,
    single_file_to_process_index=None,
    skip_if_output_files_exist=True,
):
    # Get location and paths
    location = config["location_specification"][location_key]
    std_partition_path = file_manager.get_standardized_partition_output_dir(
        config, location
    )
    vap_output_dir = Path(file_manager.get_vap_output_dir(config, location))

    # Find all input files to potentially process
    input_nc_files = sorted(list(std_partition_path.rglob("*.nc")))
    if not input_nc_files:
        print("No input .nc files found to process.")
        return

    # Check if we should skip processing entirely
    if skip_if_output_files_exist:
        existing_output_files = sorted(list(vap_output_dir.rglob("*.nc")))
        if len(existing_output_files) >= 12:
            print(
                f"Found {len(existing_output_files)} files in {vap_output_dir}. Skipping derive vap!"
            )
            return

    # Determine which files need processing
    files_to_process = []

    if skip_if_output_files_exist:
        # Get indices of files that have already been processed
        processed_indices = get_processed_file_indices(vap_output_dir)

        # Only add files that haven't been processed yet
        for file_index, nc_file in enumerate(input_nc_files, start=1):
            if file_index not in processed_indices:
                files_to_process.append((nc_file, file_index))
    else:
        # Process all files
        files_to_process = [
            (nc_file, idx) for idx, nc_file in enumerate(input_nc_files, start=1)
        ]

    if not files_to_process:
        print("No new files to process.")
        return

    # Handle single file processing if specified
    if single_file_to_process_index is not None:
        if single_file_to_process_index < len(files_to_process):
            files_to_process = [files_to_process[single_file_to_process_index]]
        else:
            print(f"Single file index {single_file_to_process_index} out of range.")
            return

    # Process the files
    total_files = len(files_to_process)
    print(f"Processing {total_files} vap data files sequentially")

    results = []
    overall_start_time = time.time()

    for i, (nc_file, file_index) in enumerate(files_to_process, 1):
        print(f"Processing file {i}/{total_files}: {nc_file}")
        result = process_single_file(
            nc_file,
            config,
            location,
            surface_elevation_offset_path,
            vap_output_dir,
            file_index,
            total_files=total_files,
            start_time=overall_start_time,
        )
        results.append(result)

        # Force garbage collection after each file
        gc.collect()

    # Calculate total elapsed time
    total_elapsed_time = time.time() - overall_start_time
    total_hours, remainder = divmod(total_elapsed_time, 3600)
    total_minutes, total_seconds = divmod(remainder, 60)

    print(f"Completed processing {len(results)} files sequentially.")
    print(
        f"Total processing time: {int(total_hours):02d}:{int(total_minutes):02d}:{int(total_seconds):02d} (hh:mm:ss)"
    )
    print(f"Average time per file: {total_elapsed_time / len(results):.2f} seconds")

    # Check for any errors in processing (for both approaches)
    failed_files = [i for i in results if i < 0]
    if failed_files:
        print(
            f"WARNING: {len(failed_files)} files failed to process: {[-i for i in failed_files]}"
        )
