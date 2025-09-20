"""
Distance to Shore Calculation Manager

This module provides a class for calculating distance to shore using high-resolution
GSHHG (Global Self-consistent, Hierarchical, High-resolution Geography) coastline data.
"""

import time
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from .deps_manager import DependencyManager


class DistanceToShoreCalculator:
    """
    Calculate distance to shore using GSHHG full resolution coastline data.

    This class downloads and manages GSHHG data, then provides methods to calculate
    the closest distance to shore from lat/lon coordinates. Distance is set to 0
    for points that are onshore.

    Attributes:
        units (str): Output units - 'nautical_miles' or 'kilometers'
        coastline_gdf (gpd.GeoDataFrame): Loaded coastline data
        coastline_projected (gpd.GeoDataFrame): Coastline data in projected CRS
        coastline_sindex: Spatial index for efficient queries
    """

    def __init__(self, config, units="nautical_miles"):
        """
        Initialize the distance to shore calculator.

        Parameters:
            config (dict): Configuration dictionary containing dependencies and directories
            units (str): Output units - 'nautical_miles' or 'kilometers'
        """
        self.config = config
        self.units = units
        self.coastline_gdf = None
        self.coastline_projected = None
        self.coastline_sindex = None

        # Validate units
        if units not in ["nautical_miles", "kilometers"]:
            raise ValueError("Units must be 'nautical_miles' or 'kilometers'")

        # Initialize dependency manager
        self.deps_manager = DependencyManager(config)

        # Download and load data
        self._download_dependencies()
        self._load_coastline_data()

    def _download_dependencies(self):
        """Download GSHHG data if not already present."""
        self.gshhg_dir = self.deps_manager.download_dependency("uh_gshhg")

    def _load_coastline_data(self):
        """Load and prepare GSHHG full resolution coastline data."""
        # GSHHG full resolution coastline file (prefer .gpkg over .shp)
        try:
            coastline_path = self.deps_manager.find_gis_file("uh_gshhg", "GSHHS_f_L1")
        except FileNotFoundError:
            # Fallback to the traditional path structure
            gshhg_dir = self.deps_manager.get_dependency_path("uh_gshhg")
            coastline_path = gshhg_dir / "GSHHS_shp" / "f" / "GSHHS_f_L1.shp"

            if not coastline_path.exists():
                raise FileNotFoundError(
                    f"GSHHG coastline file not found. Tried:\n"
                    f"- Auto-search for GSHHS_f_L1.gpkg/shp in {gshhg_dir}\n"
                    f"- Traditional path: {coastline_path}\n"
                    "Please check the GSHHG data extraction."
                )

        print(f"Loading GSHHG full resolution coastline data from {coastline_path}...")
        start_time = time.time()

        # Load coastline data
        self.coastline_gdf = gpd.read_file(coastline_path)
        print(f"Loaded coastline data: {len(self.coastline_gdf)} features")

        # Transform to appropriate projection for distance calculations
        # Using EPSG:4087 (World Equidistant Cylindrical) for consistency with existing code
        print("Projecting coastline data to EPSG:4087...")
        self.coastline_projected = self.coastline_gdf.to_crs("EPSG:4087")

        # Create spatial index for efficiency
        print("Creating spatial index...")
        self.coastline_sindex = self.coastline_projected.sindex

        # Prepare coastline geometries for distance calculations
        self._prepare_coastline_geometries()

        load_time = time.time() - start_time
        print(f"Coastline data loaded and prepared in {load_time:.2f} seconds")

    def _prepare_coastline_geometries(self):
        """
        Prepare coastline geometries for efficient distance calculations.

        This method converts non-lineal geometries (polygons) to their boundaries
        to ensure compatibility with projection operations needed for finding
        closest points on coastlines.
        """
        print("Preparing coastline geometries for distance calculations...")

        # Create a copy for prepared geometries
        self.coastline_prepared = self.coastline_projected.copy()

        # Handle non-lineal geometries by converting to boundaries
        mask_polygon = self.coastline_prepared.geometry.geom_type.isin(
            ["Polygon", "MultiPolygon"]
        )
        if mask_polygon.any():
            print(f"Converting {mask_polygon.sum()} polygon features to boundaries...")
            self.coastline_prepared.loc[mask_polygon, "geometry"] = (
                self.coastline_prepared.loc[mask_polygon, "geometry"].boundary
            )

        # Create spatial index for the prepared geometries
        self.coastline_prepared_sindex = self.coastline_prepared.sindex

        print("Coastline geometry preparation complete.")

    def calc_distance_to_shore(self, df):
        """
        Calculate distance to shore for DataFrame with lat/lon coordinates.

        Parameters:
            df (pd.DataFrame): DataFrame with 'latitude' and 'longitude' columns

        Returns:
            pd.DataFrame: Original DataFrame with added columns:
                - 'distance_to_shore': Distance to shore in specified units (float32)
                - 'closest_shore_lat': Latitude of closest point on coastline (float32)
                - 'closest_shore_lon': Longitude of closest point on coastline (float32)
        """
        if self.coastline_gdf is None:
            raise RuntimeError(
                "Coastline data not loaded. Call _load_coastline_data() first."
            )

        print(f"Calculating distance to shore for {len(df)} coordinates...")
        start_time = time.time()

        # Create GeoDataFrame from coordinates
        geometry = [
            Point(float(lon), float(lat))
            for lat, lon in zip(df["latitude_center"], df["longitude_center"])
        ]
        points_gdf = gpd.GeoDataFrame(
            df[["latitude_center", "longitude_center"]],
            geometry=geometry,
            crs="EPSG:4326",
        )

        # Transform to projected coordinate system
        points_projected = points_gdf.to_crs("EPSG:4087")

        distances = []
        closest_shore_lats = []
        closest_shore_lons = []

        for idx, point_row in points_projected.iterrows():
            point_geom = point_row.geometry

            # Find potential intersections using spatial index of prepared geometries
            possible_matches_index = list(
                self.coastline_prepared_sindex.intersection(point_geom.bounds)
            )
            buffer_kilometers = 500
            buffer_meters = buffer_kilometers * 1000

            if len(possible_matches_index) == 0:
                # Expand search area if no immediate matches
                # buffer = point_geom.buffer(100000)  # 100km buffer
                buffer = point_geom.buffer(buffer_meters)  # 100km buffer
                possible_matches_index = list(
                    self.coastline_prepared_sindex.intersection(buffer.bounds)
                )

            if len(possible_matches_index) == 0:
                raise ValueError(
                    f"No coastline features found {buffer_kilometers}km from point at index {idx} "
                    f"({point_row['latitude_center']:.4f}, {point_row['longitude_center']:.4f}). "
                    "This suggests incomplete coastline data coverage."
                )

            # Calculate distances and find minimum using prepared geometries
            possible_matches = self.coastline_prepared.iloc[possible_matches_index]
            distance_values = possible_matches.geometry.distance(point_geom)
            distance_m = distance_values.min()

            # Find the closest coastline feature and get the closest point on it
            closest_idx = distance_values.idxmin()
            closest_coastline = possible_matches.loc[closest_idx, "geometry"]

            # Handle MultiLineString by finding closest component
            if closest_coastline.geom_type == "MultiLineString":
                min_dist = float("inf")
                closest_line = None
                for line in closest_coastline.geoms:
                    line_dist = line.distance(point_geom)
                    if line_dist < min_dist:
                        min_dist = line_dist
                        closest_line = line
                closest_coastline = closest_line

            # Project point onto closest coastline and interpolate
            projection_distance = closest_coastline.project(point_geom)
            # Ensure scalar value to avoid NumPy deprecation warnings
            projection_distance = float(projection_distance)
            closest_shore_point = closest_coastline.interpolate(projection_distance)

            # Convert distance based on units
            if self.units == "nautical_miles":
                distance_output = distance_m / 1852.0  # meters to nautical miles
            else:  # kilometers
                distance_output = distance_m / 1000.0  # meters to kilometers

            # Handle points on land (very small distances)
            # Use more reasonable threshold: 1.0 NM for nautical miles, 2.0 km for kilometers
            # This accounts for coastal cities and GSHHG resolution limitations
            if distance_output < (1.0 if self.units == "nautical_miles" else 2.0):
                distance_output = 0.0
                # For points on land, use the face center coordinates as "closest shore point"
                closest_shore_point = point_geom

            distances.append(distance_output)

            # Convert closest shore point back to WGS84 for storage
            closest_shore_point_wgs84 = gpd.GeoSeries(
                [closest_shore_point], crs="EPSG:4087"
            ).to_crs("EPSG:4326")[0]

            # Extract scalar values to avoid NumPy deprecation warnings
            lat_value = float(closest_shore_point_wgs84.y)
            lon_value = float(closest_shore_point_wgs84.x)
            closest_shore_lats.append(lat_value)
            closest_shore_lons.append(lon_value)

        # Add columns to DataFrame
        result_df = df.copy()
        result_df["distance_to_shore"] = pd.Series(
            distances, dtype=np.float32, index=df.index
        )
        result_df["closest_shore_lat"] = pd.Series(
            closest_shore_lats, dtype=np.float32, index=df.index
        )
        result_df["closest_shore_lon"] = pd.Series(
            closest_shore_lons, dtype=np.float32, index=df.index
        )

        calc_time = time.time() - start_time
        units_label = "NM" if self.units == "nautical_miles" else "km"
        print(
            f"Distance calculation complete in {calc_time:.2f} seconds. "
            f"Stats: min={result_df['distance_to_shore'].min():.3f}, "
            f"max={result_df['distance_to_shore'].max():.3f}, "
            f"mean={result_df['distance_to_shore'].mean():.3f} {units_label}"
        )

        return result_df

    def get_metadata(self):
        """
        Get metadata about the distance to shore calculation.

        Returns:
            dict: Metadata dictionary
        """
        units_label = (
            "nautical_miles" if self.units == "nautical_miles" else "kilometers"
        )
        units_symbol = "NM" if self.units == "nautical_miles" else "km"

        return {
            "variable_name": "distance_to_shore",
            "standard_name": "distance_to_shore",
            "long_name": "Distance to Nearest Shore",
            "units": units_label,
            "units_symbol": units_symbol,
            "description": f"Geodesic distance from coordinate to nearest coastline in {units_label}",
            "computation": "Spatial distance calculation using GSHHG full resolution coastline data",
            "projection": "EPSG:4087 (World Equidistant Cylindrical)",
            "minimum_distance": 0.0,
            "land_points_distance": 0.0,
            "dtype": "float32",
            "data_source": "Global Self-consistent, Hierarchical, High-resolution Geography (GSHHG) v2.3.7",
            "data_source_url": "https://www.soest.hawaii.edu/pwessel/gshhg/",
            "resolution": "full",
            "creation_date": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "additional_columns": {
                "closest_shore_lat": "Latitude of closest point on coastline (WGS84)",
                "closest_shore_lon": "Longitude of closest point on coastline (WGS84)",
            },
        }
