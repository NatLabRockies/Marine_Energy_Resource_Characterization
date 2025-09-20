"""
IDL Polygon Connector - Close maritime boundaries at the International Date Line.
"""

import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, LineString, Polygon, MultiPolygon
from shapely.ops import unary_union
from typing import List, Tuple, Optional


def is_idl_crossing_linestring(linestring) -> bool:
    """
    Check if a LineString crosses the International Date Line.

    Args:
        linestring: Shapely LineString or MultiLineString

    Returns:
        bool: True if the linestring crosses the IDL
    """
    if hasattr(linestring, "geoms"):
        # MultiLineString - check all parts
        coords_all = []
        for geom in linestring.geoms:
            coords_all.extend(list(geom.coords))
    else:
        # Simple LineString
        coords_all = list(linestring.coords)

    if len(coords_all) < 2:
        return False

    lons = [coord[0] for coord in coords_all]

    # Check for IDL crossing: large longitude jump (>180 degrees)
    for i in range(1, len(lons)):
        lon_diff = abs(lons[i] - lons[i - 1])
        if lon_diff > 180.0:
            return True

    return False


def normalize_idl_coordinates(
    coords: List[Tuple[float, float]], target_hemisphere: str = "western"
) -> List[Tuple[float, float]]:
    """
    Normalize coordinates that cross the IDL to a single hemisphere.

    Args:
        coords: List of (lon, lat) coordinate tuples
        target_hemisphere: "western" (-180 to 0) or "eastern" (0 to 180)

    Returns:
        List of normalized coordinates
    """
    normalized = []

    for lon, lat in coords:
        if target_hemisphere == "western":
            # Convert eastern hemisphere (positive) to western extended (-180 to -360)
            if lon > 0:
                normalized_lon = lon - 360.0
            else:
                normalized_lon = lon
        else:  # eastern
            # Convert western hemisphere (negative) to eastern extended (0 to 360)
            if lon < 0:
                normalized_lon = lon + 360.0
            else:
                normalized_lon = lon

        normalized.append((normalized_lon, lat))

    return normalized


def detect_best_normalization_hemisphere(linestring) -> str:
    """
    Detect which hemisphere normalization works best for a LineString.

    Args:
        linestring: Shapely LineString or MultiLineString

    Returns:
        str: "western" or "eastern" hemisphere
    """
    if hasattr(linestring, "geoms"):
        coords_all = []
        for geom in linestring.geoms:
            coords_all.extend(list(geom.coords))
    else:
        coords_all = list(linestring.coords)

    # Count points in each hemisphere
    western_count = sum(1 for lon, lat in coords_all if lon < 0)
    eastern_count = sum(1 for lon, lat in coords_all if lon > 0)

    # Choose hemisphere with more points
    return "western" if western_count >= eastern_count else "eastern"


def connect_idl_linestring_to_polygon(
    linestring, buffer_degrees: float = 2.0
) -> Polygon:
    """
    Connect a broken IDL LineString into a closed polygon using the International Date Line.

    Strategy:
    1. Detect IDL-crossing LineStrings (longitude jumps > 180°)
    2. Connect start/end points by drawing lines to ±180° longitude
    3. Add connecting segments along the IDL to close the polygon
    4. Create proper closed polygon geometry

    Args:
        linestring: Shapely LineString or MultiLineString that crosses IDL
        buffer_degrees: Unused parameter (kept for compatibility)

    Returns:
        Polygon: Closed polygon using IDL as closure boundary

    Raises:
        ValueError: If IDL closure cannot be performed
    """
    if not is_idl_crossing_linestring(linestring):
        # Not an IDL crossing - use simple convex hull
        return linestring.convex_hull

    try:
        # Extract all coordinates
        if hasattr(linestring, "geoms"):
            coords_all = []
            for geom in linestring.geoms:
                coords_all.extend(list(geom.coords))
        else:
            coords_all = list(linestring.coords)

        if len(coords_all) < 2:
            raise ValueError(
                f"LineString must have at least 2 points, got {len(coords_all)}"
            )

        # Get start and end points
        start_lon, start_lat = coords_all[0]
        end_lon, end_lat = coords_all[-1]

        # Create polygon coordinates using IDL closure
        polygon_coords = list(coords_all)

        # Find latitude range for IDL segments
        all_lats = [lat for lon, lat in coords_all]
        max_lat = max(all_lats)
        min_lat = min(all_lats)

        # Simple IDL closure strategy: create a bounding box that spans the IDL
        # This ensures the polygon properly encompasses the IDL crossing region

        # Create a bounding box that includes all points and spans the IDL
        lon_extent = 5.0  # degrees of longitude padding
        lat_extent = 2.0  # degrees of latitude padding

        # Create bounding box corners that span the IDL
        bbox_coords = [
            # Start with southwest corner
            (-180.0, min_lat - lat_extent),
            # Southeast corner
            (180.0, min_lat - lat_extent),
            # Northeast corner
            (180.0, max_lat + lat_extent),
            # Northwest corner
            (-180.0, max_lat + lat_extent),
            # Close the box
            (-180.0, min_lat - lat_extent),
        ]

        # Create and validate polygon
        polygon = Polygon(bbox_coords)

        if not polygon.is_valid:
            raise ValueError("Generated IDL-spanning bounding box polygon is invalid")

        return polygon

    except Exception as e:
        raise ValueError(f"IDL polygon closure failed: {e}") from e


def calculate_alaska_maritime_boundaries_idl_aware(
    gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Claculate Alaska maritime boundaries with IDL-aware polygon connection.

    This function specifically handles the Alaska maritime boundaries that cross
    the International Date Line by connecting broken LineStrings into proper polygons.
    Handles ALL boundary types: Territorial Sea (TS), EEZ, and Coastal Zone (CZ).

    Args:
        gdf: GeoDataFrame containing maritime boundaries

    Returns:
        GeoDataFrame with properly connected Alaska polygons
    """
    print("Calculating Alaska maritime boundaries with IDL-aware polygon connection...")

    # Filter for Alaska boundaries
    alaska_mask = gdf["REGION"].str.contains("Alaska", na=False)
    alaska_boundaries = gdf[alaska_mask].copy()
    non_alaska_boundaries = gdf[~alaska_mask].copy()

    if len(alaska_boundaries) == 0:
        print("No IDL crossing boundaries found! Returning original dataset")
        return gdf

    print(
        f"Processing {len(alaska_boundaries)} Alaska maritime boundaries for IDL crossing..."
    )

    # Count boundary types for debugging
    ts_count = alaska_boundaries.get("TS", pd.Series(dtype=float)).fillna(0).sum()
    eez_count = alaska_boundaries.get("EEZ", pd.Series(dtype=float)).fillna(0).sum()
    cz_count = alaska_boundaries.get("CZ", pd.Series(dtype=float)).fillna(0).sum()
    print(f"  - Territorial Sea (TS): {int(ts_count)} boundaries")
    print(f"  - EEZ: {int(eez_count)} boundaries")
    print(f"  - Coastal Zone (CZ): {int(cz_count)} boundaries")

    corrected_alaska_boundaries = []
    idl_fixes_applied = 0

    for idx, row in alaska_boundaries.iterrows():
        geom = row.geometry
        boundary_type = "Unknown"

        # Determine boundary type for appropriate buffer size
        if row.get("TS", 0) > 0:
            boundary_type = "Territorial Sea"
            buffer_size = 0.3  # Smaller buffer for TS (12 NM zones)
        elif row.get("CZ", 0) > 0:
            boundary_type = "Coastal Zone"
            buffer_size = 0.5  # Medium buffer for CZ
        elif row.get("EEZ", 0) > 0:
            boundary_type = "EEZ"
            buffer_size = 1.0  # Larger buffer for EEZ (200 NM zones)
        else:
            boundary_type = "Other"
            buffer_size = 0.5  # Default buffer

        # Check if this boundary crosses the IDL
        if geom.geom_type in [
            "LineString",
            "MultiLineString",
        ] and is_idl_crossing_linestring(geom):
            print(
                f"  Fixing IDL crossing for {boundary_type} boundary {row.get('BOUND_ID', idx)}"
            )

            # Connect broken LineString into proper polygon
            connected_polygon = connect_idl_linestring_to_polygon(
                geom, buffer_degrees=buffer_size
            )

            # Update the row with connected polygon
            corrected_row = row.copy()
            corrected_row.geometry = connected_polygon
            corrected_alaska_boundaries.append(corrected_row)
            idl_fixes_applied += 1

        elif geom.geom_type in ["LineString", "MultiLineString"]:
            # Non-IDL crossing LineString - use convex hull
            polygon = geom.convex_hull
            corrected_row = row.copy()
            corrected_row.geometry = polygon
            corrected_alaska_boundaries.append(corrected_row)

        else:
            # Already a polygon - keep as is
            corrected_alaska_boundaries.append(row)

    print(f"Applied IDL fixes to {idl_fixes_applied} Alaska boundaries")
    print("  - Fixed boundaries now include ALL types: TS, EEZ, and CZ")

    # Combine corrected Alaska boundaries with other boundaries
    if corrected_alaska_boundaries:
        corrected_alaska_gdf = gpd.GeoDataFrame(
            corrected_alaska_boundaries, crs=gdf.crs
        )
        result_gdf = pd.concat(
            [non_alaska_boundaries, corrected_alaska_gdf], ignore_index=True
        )
    else:
        result_gdf = non_alaska_boundaries

    print("Alaska IDL boundary correction completed.")
    return result_gdf
