"""
Jurisdiction Manager

This module provides jurisdiction calculation with improved boundary handling
and specialized methods for different geographic regions.
"""

import json
import time
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from .deps_manager import DependencyManager


class JurisdictionCalculator:
    """
    Jurisdiction calculator with specialized boundary handling.

    Provides both standard jurisdiction calculation and specialized methods
    for regions requiring custom processing like IDL boundary handling.
    """

    def __init__(self, config):
        """
        Initialize the jurisdiction calculator.

        Parameters:
            config (dict): Configuration dictionary containing dependencies and directories
        """
        self.config = config

        self.noaa_czma_gdf = None
        self.noaa_eez_and_territorial_sea_gdf = None
        self.noaa_coastal_states_gdf = None

        # Initialize dependency manager
        self.deps_manager = DependencyManager(config)

        # Download and load data
        self._download_dependencies()
        self._load_jurisdiction_data()

        self.usa = "United States of America"
        self.eez = "Exclusive Economic Zone"

    def _download_dependencies(self):
        self.deps_manager.download_dependency(
            "marinecadastre_coastal_zone_management_act"
        )
        self.deps_manager.download_dependency("noaa_territorial_sea")
        self.deps_manager.download_dependency("noaa_coastal_states")

    def _load_jurisdiction_data(self):
        """Load and process jurisdiction data"""
        print("Loading jurisdiction data...")
        self._load_coastal_zone_management_act_data()
        self._load_noaa_eez_and_territorial_sea_data()
        self._load_noaa_coastal_states()
        print("Standardizing CRS to EPSG:4326...")
        self._standardize_crs()
        print("Jurisdiction data loaded and standardized.")

    def _load_coastal_zone_management_act_data(self):
        czma_path = self.deps_manager.find_gis_file(
            "marinecadastre_coastal_zone_management_act", "CoastalZoneManagementAct"
        )
        self.noaa_czma_gdf = gpd.read_file(czma_path)

    def _load_noaa_eez_and_territorial_sea_data(self):
        noaa_eez_and_territorial_sea_path = self.deps_manager.find_gis_file(
            "noaa_territorial_sea", "USMaritimeLimitsNBoundaries"
        )
        self.noaa_eez_and_territorial_sea_gdf = gpd.read_file(
            noaa_eez_and_territorial_sea_path
        )

    def _load_noaa_coastal_states(self):
        noaa_coastal_states_path = self.deps_manager.find_gis_file(
            "noaa_coastal_states", "CoastalState"
        )
        self.noaa_coastal_states_gdf = gpd.read_file(noaa_coastal_states_path)

    def _standardize_crs(self):
        """Standardize all GeoDataFrames to EPSG:4326 CRS"""
        self.noaa_czma_gdf = self.noaa_czma_gdf.to_crs("EPSG:4326")
        self.noaa_eez_and_territorial_sea_gdf = (
            self.noaa_eez_and_territorial_sea_gdf.to_crs("EPSG:4326")
        )
        self.noaa_coastal_states_gdf = self.noaa_coastal_states_gdf.to_crs("EPSG:4326")

    def _find_features_containing_point(self, gdf, point_geom):
        """
        Find all rows in a geographic dataset that contain a point.

        Uses standard spatial queries since IDL boundaries are now properly connected.
        """
        try:
            # Use standard spatial index for spatial queries
            possible_matches = list(gdf.sindex.intersection(point_geom.bounds))

            if len(possible_matches) == 0:
                return None

            candidate_features = gdf.iloc[possible_matches]
            containing_features = candidate_features[
                candidate_features.geometry.contains(point_geom)
            ]

            if len(containing_features) == 0:
                return None

            return containing_features

        except Exception as e:
            print(f"Spatial query failed: {e}")
            return None

    def calc_jurisdiction(self, df):
        """
        Calculate maritime jurisdiction using standard boundary processing.

        This method processes points using standard (non-IDL) boundary conversion.
        For Aleutian Islands points that cross the IDL, use calc_aleutian_islands_jurisdiction() instead.
        """
        # Create GeoDataFrame from coordinates
        geometry = [
            Point(lon, lat)
            for lat, lon in zip(df["latitude_center"], df["longitude_center"])
        ]
        points_gdf = gpd.GeoDataFrame(
            df[["latitude_center", "longitude_center"]],
            geometry=geometry,
            crs="EPSG:4326",
        )

        # Process each point with standard boundaries
        results = []
        total_points = len(points_gdf)
        for idx, point_row in points_gdf.iterrows():
            if idx % 1000 == 0:
                print(
                    f"Processing point {idx + 1}/{total_points} ({((idx + 1) / total_points) * 100:.1f}%)"
                )

            point_geom = point_row.geometry

            # Query all data sources with standard boundaries
            query_result = self._query_all_data_sources(point_geom)
            jurisdiction_result = self._jurisdiction_lookup_table(query_result)
            results.append(jurisdiction_result)

        # Add jurisdiction columns to DataFrame
        result_df = df.copy()
        result_df["jurisdiction"] = pd.Series(
            [r["jurisdiction"] for r in results], dtype="string", index=df.index
        )
        result_df["jurisdiction_source"] = pd.Series(
            [r["jurisdiction_source"] for r in results], dtype="string", index=df.index
        )

        return result_df

    def calc_aleutian_islands_jurisdiction(self, df):
        """
        Calculate maritime jurisdiction specifically for Aleutian Islands points.

        This method applies IDL corrections to handle LineString boundaries in NOAA GIS datasets
        by converting them to closed polygons for robust point-in-polygon testing.
        Uses two approaches: IDL closure for boundaries crossing the International
        Date Line, and convex hull closure for other boundaries.

        Applies to all EEZ and TS boundaries from the NOAA
        maritime boundaries dataset.

        Parameters:
            df (pd.DataFrame): DataFrame with 'latitude', 'longitude' columns

        Returns:
            pd.DataFrame: DataFrame with jurisdiction columns using IDL closure

        Raises:
            ValueError: If IDL closure fails for any boundary
        """
        print("Calculating Aleutian Islands jurisdiction with IDL closure...")

        # Apply boundary closure to EEZ and TS boundaries (both IDL-crossing and non-IDL)
        from .idl_polygon_connector import (
            calculate_alaska_maritime_boundaries_idl_aware,
        )

        # Filter ALL EEZ and TS boundaries for correction processing
        # The correction function handles both IDL-crossing (with IDL closure) and
        # non-IDL-crossing boundaries (with convex hull closure)
        eez_and_ts_mask = (self.noaa_eez_and_territorial_sea_gdf.get("EEZ", 0) > 0) | (
            self.noaa_eez_and_territorial_sea_gdf.get("TS", 0) > 0
        )

        eez_and_ts_boundaries = self.noaa_eez_and_territorial_sea_gdf[
            eez_and_ts_mask
        ].copy()
        other_boundaries = self.noaa_eez_and_territorial_sea_gdf[
            ~eez_and_ts_mask
        ].copy()

        print(
            f"Applying IDL aware boundary to {len(eez_and_ts_boundaries)} EEZ and TS boundaries..."
        )

        # Apply specialized boundaries for Alaska/aleutian
        try:
            eez_ts_boundaries = calculate_alaska_maritime_boundaries_idl_aware(
                eez_and_ts_boundaries
            )
        except Exception as e:
            raise ValueError(
                f"Boundary correction failed for Aleutian boundaries: {e}"
            ) from e

        # Combine corrected EEZ/TS boundaries with unchanged other boundaries
        if len(other_boundaries) > 0:
            boundaries = pd.concat(
                [eez_ts_boundaries, other_boundaries], ignore_index=True
            )
        else:
            boundaries = eez_ts_boundaries

        # Temporarily replace boundaries for processing
        original_boundaries = self.noaa_eez_and_territorial_sea_gdf.copy()
        self.noaa_eez_and_territorial_sea_gdf = boundaries

        # Create GeoDataFrame from coordinates
        geometry = [
            Point(lon, lat) for lat, lon in zip(df["latitude"], df["longitude"])
        ]
        points_gdf = gpd.GeoDataFrame(
            df[["latitude", "longitude"]],
            geometry=geometry,
            crs="EPSG:4326",
        )

        # Process each point with IDL-closed boundaries
        results = []
        total_points = len(points_gdf)
        for idx, point_row in points_gdf.iterrows():
            if idx % 1000 == 0:
                print(
                    f"Processing Aleutian point {idx + 1}/{total_points} ({((idx + 1) / total_points) * 100:.1f}%)"
                )

            point_geom = point_row.geometry

            # Query all data sources with IDL-closed boundaries
            query_result = self._query_all_data_sources(point_geom)
            jurisdiction_result = self._aleutian_jurisdiction_lookup_table(query_result)
            results.append(jurisdiction_result)

        # Restore original boundaries
        self.noaa_eez_and_territorial_sea_gdf = original_boundaries

        # Add jurisdiction columns to DataFrame
        result_df = df.copy()
        result_df["jurisdiction"] = pd.Series(
            [r["jurisdiction"] for r in results], dtype="string", index=df.index
        )
        result_df["jurisdiction_source"] = pd.Series(
            [r["jurisdiction_source"] for r in results], dtype="string", index=df.index
        )

        print(
            f"Successfully processed {len(results)} Aleutian Islands points with IDL closure"
        )

        return result_df

    def _usa_coastal_state_jurisdiction_string(self, state):
        return f"{self.usa}, {state} State Waters"

    def _usa_territorial_sea_jurisdiction_string(self):
        return f"{self.usa}, Territorial Sea"

    def _usa_eez_jurisdiction_string(self):
        return f"{self.usa}, {self.eez}"

    def _outside_usa_jurisdiction_string(self):
        return f"Not Specified, outside jurisdiction of {self.usa}"

    def _aleutian_jurisdiction_lookup_table(self, query_result):
        noaa_eez_and_territorial_query = query_result["noaa_eez_and_territorial_sea"]
        in_noaa_territorial_sea = noaa_eez_and_territorial_query["territorial_sea"]
        in_noaa_eez = noaa_eez_and_territorial_query["eez"]

        noaa_coastal_state_query = query_result["coastal_states"]
        in_noaa_coastal_states = noaa_coastal_state_query[
            "point_within_coastal_states_boundaries"
        ]
        noaa_coastal_state = noaa_coastal_state_query["state_name"]

        if in_noaa_coastal_states:
            return {
                "jurisdiction": self._usa_coastal_state_jurisdiction_string(
                    noaa_coastal_state
                ),
                "jurisdiction_source": self.config["dependencies"]["gis"][
                    "noaa_coastal_states"
                ]["docs"],
            }
        elif in_noaa_territorial_sea:
            return {
                "jurisdiction": self._usa_territorial_sea_jurisdiction_string(),
                "jurisdiction_source": self.config["dependencies"]["gis"][
                    "noaa_territorial_sea"
                ]["docs"],
            }
        elif in_noaa_eez:
            return {
                "jurisdiction": self._usa_eez_jurisdiction_string(),
                "jurisdiction_source": self.config["dependencies"]["gis"]["noaa_eez"][
                    "docs"
                ],
            }
        else:
            return {
                "jurisdiction": self._outside_usa_jurisdiction_string(),
                "jurisdiction_source": "None",
            }

    def _query_all_data_sources(self, point_geom):
        """Query all data sources for jurisdiction determination"""
        results = {
            "czma": self._query_czma(point_geom),
            "noaa_eez_and_territorial_sea": self._query_noaa_eez_and_territorial_sea(
                point_geom
            ),
            "coastal_states": self._query_coastal_states(point_geom),
        }
        return results

    def _query_czma(self, point_geom):
        containing_features = self._find_features_containing_point(
            self.noaa_czma_gdf, point_geom
        )

        if containing_features is None:
            return {"point_within_czma_boundary": False, "czma_point_classifier": None}

        czma_classifiers = containing_features["CZMADomain"].tolist()

        if len(czma_classifiers) == 1:
            return {
                "point_within_czma_boundary": True,
                "czma_point_classifier": czma_classifiers[0],
            }
        else:
            return {
                "point_within_czma_boundary": True,
                "czma_point_classifier": czma_classifiers,
            }

    def _query_noaa_eez_and_territorial_sea(self, point_geom):
        containing_features = self._find_features_containing_point(
            self.noaa_eez_and_territorial_sea_gdf, point_geom
        )

        result = {
            "point_within_noaa_eez_and_territorial_sea_bounds": False,
            "territorial_sea": False,
            "coastal_zone": False,
            "eez": False,
            "f_eez": False,
            "region": None,
        }

        if containing_features is None:
            return result

        # Process containing features with hierarchical priority
        combined_result = {
            "point_within_noaa_eez_and_territorial_sea_bounds": True,
            "territorial_sea": False,
            "coastal_zone": False,
            "eez": False,
            "f_eez": False,
            "region": [],
        }

        # Hierarchical priority based on maritime distance zones:
        # This follows the principle that more restrictive/closer zones take precedence
        if any(row["TS"] > 0.0 for _, row in containing_features.iterrows()):
            combined_result["territorial_sea"] = True
        elif any(row["EEZ"] > 0.0 for _, row in containing_features.iterrows()):
            combined_result["eez"] = True
        elif any(row["CZ"] > 0.0 for _, row in containing_features.iterrows()):
            combined_result["coastal_zone"] = True

        # F_EEZ (Federal EEZ) can coexist with other classifications
        if any(row["F_EEZ"] > 0.0 for _, row in containing_features.iterrows()):
            combined_result["f_eez"] = True

        # Collect regions
        for _, this_row in containing_features.iterrows():
            region = this_row["REGION"]
            if region not in combined_result["region"]:
                combined_result["region"].append(region)

        return combined_result

    def _query_coastal_states(self, point_geom):
        containing_states = self._find_features_containing_point(
            self.noaa_coastal_states_gdf, point_geom
        )

        if containing_states is None:
            return {"point_within_coastal_states_boundaries": False, "state_name": None}

        state_names = containing_states["stateName"].tolist()

        if len(state_names) == 1:
            return {
                "point_within_coastal_states_boundaries": True,
                "state_name": state_names[0],
            }
        else:
            raise ValueError(
                f"Point {point_geom} within multiple coastal states: {state_names}"
            )

    def _jurisdiction_lookup_table(self, query_result):
        """
        Jurisdiction lookup table with distance-based validation for remote points.
        """
        # Extract query results
        czma_query = query_result["czma"]
        in_coastal_zone_management_act_dataset = (
            czma_query["point_within_czma_boundary"] is True
        )
        coastal_zone_management_act_classifier = czma_query["czma_point_classifier"]

        noaa_eez_and_territorial_query = query_result["noaa_eez_and_territorial_sea"]
        in_noaa_eez_and_territorial_sea_bounds = noaa_eez_and_territorial_query[
            "point_within_noaa_eez_and_territorial_sea_bounds"
        ]
        in_noaa_territorial_sea = noaa_eez_and_territorial_query["territorial_sea"]
        in_noaa_coastal_zone = noaa_eez_and_territorial_query["coastal_zone"]
        in_noaa_eez = noaa_eez_and_territorial_query["eez"]
        in_noaa_f_eez = noaa_eez_and_territorial_query["f_eez"]
        noaa_regions = noaa_eez_and_territorial_query["region"]

        noaa_coastal_state_query = query_result["coastal_states"]
        in_noaa_coastal_states = noaa_coastal_state_query[
            "point_within_coastal_states_boundaries"
        ]
        noaa_coastal_state = noaa_coastal_state_query["state_name"]

        # If a point is in disputed/remote areas but not in a US coastal state,
        # apply additional validation to prevent incorrect US jurisdiction claims
        def is_remote_disputed_area():
            """Check if point is in a remote area that might have disputed US claims."""
            # If point is detected as US maritime zone but NOT in any US coastal state,
            # and the region suggests it's near international borders, be more conservative
            if not in_noaa_coastal_states and noaa_regions:
                # Atlantic Coast regions near Canada (Gulf of Maine, Bay of Fundy area)
                if "Atlantic Coast" in str(noaa_regions):
                    return True
                # Pacific regions that might extend too far
                if "Pacific Coast" in str(noaa_regions) and not in_noaa_coastal_states:
                    # Only apply to territorial sea claims, not EEZ
                    return in_noaa_territorial_sea
            return False

        # Standard jurisdiction determination logic with validation
        if in_noaa_coastal_states:
            return {
                "jurisdiction": self._usa_coastal_state_jurisdiction_string(
                    noaa_coastal_state
                ),
                "jurisdiction_source": self.config["dependencies"]["gis"][
                    "noaa_coastal_states"
                ]["docs"],
            }
        elif in_noaa_territorial_sea:
            if is_remote_disputed_area():
                # Conservative: Don't claim territorial sea for remote disputed areas
                return {
                    "jurisdiction": self._outside_usa_jurisdiction_string(),
                    "jurisdiction_source": "None",
                }
            else:
                return {
                    "jurisdiction": self._usa_territorial_sea_jurisdiction_string(),
                    "jurisdiction_source": self.config["dependencies"]["gis"][
                        "noaa_territorial_sea"
                    ]["docs"],
                }
        elif in_noaa_eez:
            return {
                "jurisdiction": self._usa_eez_jurisdiction_string(),
                "jurisdiction_source": self.config["dependencies"]["gis"]["noaa_eez"][
                    "docs"
                ],
            }
        elif in_coastal_zone_management_act_dataset:
            if "federal consistency" in coastal_zone_management_act_classifier:
                return {
                    "jurisdiction": self._usa_eez_jurisdiction_string(),
                    "jurisdiction_source": self.config["dependencies"]["gis"][
                        "marinecadastre_coastal_zone_management_act"
                    ]["docs"],
                }
        elif noaa_regions and "Alaska" in noaa_regions:
            return {
                "jurisdiction": self._usa_eez_jurisdiction_string(),
                "jurisdiction_source": self.config["dependencies"]["gis"]["noaa_eez"][
                    "docs"
                ],
            }
        elif noaa_coastal_state == "Alaska":
            return {
                "jurisdiction": self._usa_eez_jurisdiction_string(),
                "jurisdiction_source": self.config["dependencies"]["gis"]["noaa_eez"][
                    "docs"
                ],
            }
        else:
            return {
                "jurisdiction": self._outside_usa_jurisdiction_string(),
                "jurisdiction_source": "None",
            }

    def get_metadata(self):
        """
        Get comprehensive metadata for jurisdiction calculation.

        Returns
        -------
        dict
            Metadata dictionary with CF-compliant attributes and data source information
        """

        # Get dependency information
        czma_info = self.deps_manager.get_dependency_info(
            "marinecadastre_coastal_zone_management_act"
        )
        territorial_info = self.deps_manager.get_dependency_info("noaa_territorial_sea")
        coastal_states_info = self.deps_manager.get_dependency_info(
            "noaa_coastal_states"
        )

        metadata = {
            "variable_name": "vap_jurisdiction",
            "long_name": "Maritime Jurisdiction Classification",
            "description": (
                f"Maritime jurisdiction classification derived from NOAA datasets: "
                f"Coastal Zone Management Act documentation: {czma_info['docs_url']}, dataset_link: {czma_info['data_url']}; "
                f"Territorial Sea documentation: {territorial_info['docs_url']}, dataset_link: {territorial_info['data_url']}; "
                f"Coastal States documentation: {coastal_states_info['docs_url']}, dataset_link: {coastal_states_info['data_url']}"
            ),
            "coverage_content_type": "auxiliaryInformation",
            "input_variables": "lat_center, lon_center coordinates for each face",
            "methodology": (
                "Geographic coordinates for each face center were spatially intersected with "
                "boundary polygons to determine jurisdictional classification."
            ),
            "coordinates": "face lat_center lon_center",
        }

        return metadata
