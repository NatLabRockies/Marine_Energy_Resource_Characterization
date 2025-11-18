"""
Distance to Shore Calculation Manager

This module provides a class for calculating distance to shore using high-resolution
GSHHG (Global Self-consistent, Hierarchical, High-resolution Geography) coastline data.
"""

import time
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from .deps_manager import DependencyManager

# Silence NumPy scalar conversion deprecation warnings from coordinate transformations
# This catches warnings from both pyproj and geopandas coordinate operations
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message=".*Conversion of an array with ndim > 0 to a scalar.*",
)
# Also catch it from pyproj specifically
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    module="pyproj",
)


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
        self.batch_size = 50000

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

    def _process_batch_vectorized(self, batch_df, batch_indices, buffer_km):
        """
        Process a batch of points using vectorized operations.

        Parameters:
            batch_df (pd.DataFrame): Batch of coordinates
            batch_indices (pd.Index): Original indices for the batch
            buffer_km (float): Search buffer in kilometers

        Returns:
            tuple: (distances, closest_lats, closest_lons) as numpy arrays
        """
        # Create GeoDataFrame for batch
        geometry = [
            Point(float(lon), float(lat))
            for lat, lon in zip(
                batch_df["latitude_center"].values, batch_df["longitude_center"].values
            )
        ]
        points_gdf = gpd.GeoDataFrame(
            batch_df[["latitude_center", "longitude_center"]],
            geometry=geometry,
            crs="EPSG:4326",
            index=batch_indices,
        )

        # Transform to projected coordinate system
        points_projected = points_gdf.to_crs("EPSG:4087")

        # Progressive buffer expansion for finding coastline features
        buffer_distances = [
            buffer_km,
            buffer_km * 5,
            buffer_km * 10,
        ]  # 100/500/1000km or 500/2500/5000km
        buffer_meters = [dist * 1000 for dist in buffer_distances]

        distances = np.full(len(batch_df), np.nan, dtype=np.float32)
        closest_shore_lats = np.full(len(batch_df), np.nan, dtype=np.float32)
        closest_shore_lons = np.full(len(batch_df), np.nan, dtype=np.float32)

        # Track which points still need processing
        unprocessed_mask = np.ones(len(batch_df), dtype=bool)

        for buffer_idx, buffer_m in enumerate(buffer_meters):
            if not unprocessed_mask.any():
                break

            current_points = points_projected[unprocessed_mask]
            if len(current_points) == 0:
                break

            print(
                f"  Buffer {buffer_distances[buffer_idx]:.0f}km: processing {len(current_points)} points"
            )

            # Create buffer around all points for spatial intersection
            buffered_points = current_points.geometry.buffer(buffer_m)

            # Find all coastline features that intersect with any buffered point
            combined_buffer = buffered_points.union_all()
            possible_coastline_indices = list(
                self.coastline_prepared_sindex.intersection(combined_buffer.bounds)
            )

            if len(possible_coastline_indices) == 0:
                continue

            relevant_coastline = self.coastline_prepared.iloc[
                possible_coastline_indices
            ]

            # Process each point in current batch
            for local_idx, (orig_idx, point_row) in enumerate(
                current_points.iterrows()
            ):
                if not unprocessed_mask[points_projected.index.get_loc(orig_idx)]:
                    continue

                point_geom = point_row.geometry

                # Find coastline features within buffer of this specific point
                point_buffer = point_geom.buffer(buffer_m)
                nearby_indices = list(
                    relevant_coastline.sindex.intersection(point_buffer.bounds)
                )

                if len(nearby_indices) == 0:
                    continue

                nearby_coastline = relevant_coastline.iloc[nearby_indices]

                # Vectorized distance calculation to all nearby coastline features
                distance_values = nearby_coastline.geometry.distance(point_geom)
                distance_m = distance_values.min()

                # Find closest coastline feature and calculate closest point
                closest_idx = distance_values.idxmin()
                closest_coastline = nearby_coastline.loc[closest_idx, "geometry"]

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
                projection_distance = float(projection_distance)
                closest_shore_point = closest_coastline.interpolate(projection_distance)

                # Convert distance based on units
                if self.units == "nautical_miles":
                    distance_output = distance_m / 1852.0
                else:
                    distance_output = distance_m / 1000.0

                # Handle points on land
                if distance_output < (1.0 if self.units == "nautical_miles" else 2.0):
                    distance_output = 0.0
                    closest_shore_point = point_geom

                # Convert closest shore point back to WGS84
                closest_shore_point_wgs84 = gpd.GeoSeries(
                    [closest_shore_point], crs="EPSG:4087"
                ).to_crs("EPSG:4326")[0]

                # Store results
                result_idx = points_projected.index.get_loc(orig_idx)
                distances[result_idx] = float(distance_output)
                # Ensure scalar extraction for coordinate values to avoid NumPy deprecation warnings
                lat_coord = closest_shore_point_wgs84.y
                lon_coord = closest_shore_point_wgs84.x
                closest_shore_lats[result_idx] = float(
                    lat_coord.item() if hasattr(lat_coord, "item") else lat_coord
                )
                closest_shore_lons[result_idx] = float(
                    lon_coord.item() if hasattr(lon_coord, "item") else lon_coord
                )

                # Mark as processed
                unprocessed_mask[result_idx] = False

        # Check for any unprocessed points
        if unprocessed_mask.any():
            unprocessed_indices = batch_indices[unprocessed_mask]
            unprocessed_coords = [
                (row["latitude_center"], row["longitude_center"])
                for _, row in batch_df[unprocessed_mask].iterrows()
            ]
            raise ValueError(
                f"Could not find coastline features within maximum buffer "
                f"({buffer_distances[-1]}km) for {unprocessed_mask.sum()} points. "
                f"Problematic coordinates: {unprocessed_coords[:5]}..."  # Show first 5
            )

        return distances, closest_shore_lats, closest_shore_lons

    def calc_distance_to_shore(self, df, buffer_km=None):
        """
        Calculate distance to shore for DataFrame with lat/lon coordinates using vectorized operations.

        Parameters:
            df (pd.DataFrame): DataFrame with 'latitude_center' and 'longitude_center' columns
            buffer_km (float, optional): Search buffer in kilometers. Auto-detected based on location if None.

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

        print(
            f"Calculating distance to shore for {len(df)} coordinates using vectorized approach..."
        )
        start_time = time.time()

        # Auto-detect buffer based on location name
        if buffer_km is None:
            location_name = str(self.config.get("location", "")).lower()
            if "aleutian" in location_name:
                buffer_km = 500
            else:
                buffer_km = 100

        print(f"Using {buffer_km}km search buffer")

        # Initialize result arrays
        distances = np.full(len(df), np.nan, dtype=np.float32)
        closest_shore_lats = np.full(len(df), np.nan, dtype=np.float32)
        closest_shore_lons = np.full(len(df), np.nan, dtype=np.float32)

        batch_size = self.batch_size

        for batch_start in range(0, len(df), batch_size):
            batch_end = min(batch_start + batch_size, len(df))
            batch_df = df.iloc[batch_start:batch_end]
            batch_indices = df.index[batch_start:batch_end]

            print(
                f"Processing batch {batch_start // batch_size + 1}/{(len(df) - 1) // batch_size + 1} "
                f"({len(batch_df)} points)..."
            )

            # Process this batch
            batch_distances, batch_lats, batch_lons = self._process_batch_vectorized(
                batch_df, batch_indices, buffer_km
            )

            # Store results
            distances[batch_start:batch_end] = batch_distances
            closest_shore_lats[batch_start:batch_end] = batch_lats
            closest_shore_lons[batch_start:batch_end] = batch_lons

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
            f"Vectorized distance calculation complete in {calc_time:.2f} seconds. "
            f"Stats: min={result_df['distance_to_shore'].min():.3f}, "
            f"max={result_df['distance_to_shore'].max():.3f}, "
            f"mean={result_df['distance_to_shore'].mean():.3f} {units_label}"
        )

        return result_df

    @staticmethod
    def get_metadata_static(config, units="nautical_miles"):
        """
        Get metadata about the distance to shore calculation without loading coastline data.

        This static method provides metadata without requiring initialization of the
        DistanceToShoreCalculator, avoiding the expensive coastline data loading.

        Parameters:
            config (dict): Configuration dictionary containing dependencies info
            units (str): Output units - 'nautical_miles' or 'kilometers'

        Returns:
            dict: Metadata dictionary
        """
        units_label = (
            "nautical_miles" if units == "nautical_miles" else "kilometers"
        )
        units_symbol = "NM" if units == "nautical_miles" else "km"

        # Get dependency information for URLs without creating DependencyManager instance
        deps_manager = DependencyManager(config)
        gshhg_info = deps_manager.get_dependency_info("uh_gshhg")

        return {
            "long_name": "Distance to Nearest Shore",
            "units": units_label,
            "units_symbol": units_symbol,
            "description": f"Geodesic distance from coordinate to nearest coastline in {units_label}. Spatial distance calculation using GSHHG full resolution coastline data with EPSG:4087 (World Equidistant Cylindrical) projection from Global Self-consistent, Hierarchical, High-resolution Geography (GSHHG) Documentation: {gshhg_info['docs_url']}, Data: {gshhg_info['data_url']}",
        }

    def get_metadata(self):
        """
        Get metadata about the distance to shore calculation.

        Returns:
            dict: Metadata dictionary
        """
        # Delegate to static method to avoid code duplication
        return self.get_metadata_static(self.config, self.units)
