"""
GIS Boundary Manager for FVCOM unstructured mesh data.

This module provides functions for:
1. Extracting the exterior boundary polygon from triangular mesh data
2. Computing grid resolution (average edge length) for each mesh element
3. Caching results to avoid recomputation

Handles special cases like the International Date Line (IDL) crossing
for the Aleutian Islands dataset.
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from pyproj import Geod
from shapely.geometry import LineString, Polygon
from shapely.ops import linemerge, polygonize

from . import file_manager


# WGS84 geodesic calculator
_geod = Geod(ellps="WGS84")


def _get_boundary_cache_path(config, location):
    """Get path to mesh boundary cache file."""
    tracking_path = file_manager.get_tracking_output_dir(config, location)
    location_config = (
        config["location_specification"][location]
        if isinstance(location, str)
        else location
    )
    return Path(
        tracking_path,
        f"{location_config['output_name']}_mesh_boundary_cache.json",
    )


def _get_grid_resolution_cache_path(config, location):
    """Get path to grid resolution cache file."""
    tracking_path = file_manager.get_tracking_output_dir(config, location)
    location_config = (
        config["location_specification"][location]
        if isinstance(location, str)
        else location
    )
    return Path(
        tracking_path,
        f"{location_config['output_name']}_grid_resolution_cache.json",
    )


def _is_aleutian_islands(location):
    """Check if location is Aleutian Islands (crosses IDL)."""
    location_config = location if isinstance(location, dict) else location
    if isinstance(location_config, dict):
        output_name = location_config.get("output_name", "")
    else:
        output_name = str(location_config)
    return "aleutian" in output_name.lower()


def _normalize_longitude_for_idl(lon_array):
    """
    Normalize longitudes for IDL-crossing datasets.

    Converts positive longitudes to extended western hemisphere
    (e.g., 170 -> -190) to avoid wrapping at -180.

    Parameters
    ----------
    lon_array : np.ndarray
        Array of longitudes in degrees

    Returns
    -------
    np.ndarray
        Normalized longitudes in extended western hemisphere
    """
    normalized = lon_array.copy()
    # Convert positive (eastern hemisphere) to extended western
    normalized[normalized > 0] = normalized[normalized > 0] - 360.0
    return normalized


def _extract_boundary_edges(nv):
    """
    Extract boundary edges from triangular mesh connectivity.

    Boundary edges are edges that appear in only one triangle.

    Parameters
    ----------
    nv : np.ndarray
        Node connectivity array, shape (3, n_elements) or (n_elements, 3)
        Contains 1-based node indices

    Returns
    -------
    list of tuple
        List of (node1, node2) tuples representing boundary edges
    """
    # Ensure shape is (n_elements, 3)
    if nv.shape[0] == 3 and nv.shape[1] != 3:
        nv = nv.T

    # Convert to 0-based indexing
    nv = nv - 1

    # Count edge occurrences
    edge_count = defaultdict(int)

    for tri in nv:
        # Create edges (sorted to ensure consistent ordering)
        edges = [
            tuple(sorted([tri[0], tri[1]])),
            tuple(sorted([tri[1], tri[2]])),
            tuple(sorted([tri[2], tri[0]])),
        ]
        for edge in edges:
            edge_count[edge] += 1

    # Boundary edges appear exactly once
    boundary_edges = [edge for edge, count in edge_count.items() if count == 1]

    return boundary_edges


def _order_boundary_edges(boundary_edges):
    """
    Order boundary edges to form a closed polygon.

    Parameters
    ----------
    boundary_edges : list of tuple
        Unordered list of (node1, node2) boundary edges

    Returns
    -------
    list of int
        Ordered list of node indices forming the exterior boundary
    """
    if not boundary_edges:
        return []

    # Build adjacency map
    adjacency = defaultdict(list)
    for n1, n2 in boundary_edges:
        adjacency[n1].append(n2)
        adjacency[n2].append(n1)

    # Start from first edge
    current = boundary_edges[0][0]
    ordered = [current]
    visited_edges = set()

    while True:
        neighbors = adjacency[current]
        next_node = None

        for neighbor in neighbors:
            edge = tuple(sorted([current, neighbor]))
            if edge not in visited_edges:
                next_node = neighbor
                visited_edges.add(edge)
                break

        if next_node is None or next_node == ordered[0]:
            break

        ordered.append(next_node)
        current = next_node

    return ordered


def compute_mesh_boundary(ds, config, location, crs_string=None):
    """
    Compute the exterior boundary polygon of an unstructured triangular mesh.

    Extracts boundary edges from the mesh connectivity and orders them to form
    a closed polygon. Handles IDL-crossing for Aleutian Islands by normalizing
    longitudes to extended western hemisphere.

    Results are cached to avoid recomputation.

    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset containing:
        - lat_node, lon_node: node coordinates
        - nv: node connectivity (3 vertices per element)
    config : dict
        Configuration dictionary
    location : str or dict
        Location key or location configuration dict
    crs_string : str, optional
        Coordinate reference system string. If None, attempts to read from
        ds.attrs['geospatial_bounds_crs']

    Returns
    -------
    dict
        Dictionary containing:
        - geospatial_bounds: WKT POLYGON string
        - geospatial_bounds_crs: CRS string
        - geospatial_lat_min/max: latitude bounds
        - geospatial_lon_min/max: longitude bounds
        - geospatial_lat_units/lon_units: coordinate units
        - geospatial_lat_resolution/lon_resolution: None for unstructured
        - boundary_coordinates: list of (lon, lat) tuples

    Raises
    ------
    ValueError
        If CRS string is not available
        If required coordinate units are missing
    """
    # Check cache first
    cache_path = _get_boundary_cache_path(config, location)
    if cache_path.exists():
        print(f"Loading cached mesh boundary from {cache_path}")
        with open(cache_path, "r") as f:
            return json.load(f)

    print("Computing mesh boundary polygon...")

    # Get coordinates
    lat_node = ds.lat_node.values
    lon_node = ds.lon_node.values
    nv = ds.nv.values

    # Check for required units
    if not hasattr(ds.lat_node, "units"):
        raise ValueError("Latitude coordinate must have units specified")
    if not hasattr(ds.lon_node, "units"):
        raise ValueError("Longitude coordinate must have units specified")

    # Get CRS
    if crs_string is not None:
        bounds_crs = crs_string
    elif "geospatial_bounds_crs" in ds.attrs:
        bounds_crs = ds.attrs["geospatial_bounds_crs"]
    else:
        raise ValueError(
            "CRS string must be provided either as input parameter or in dataset attributes"
        )

    # Normalize coordinates for IDL-crossing datasets
    is_idl = _is_aleutian_islands(location)
    if is_idl:
        print("  Normalizing coordinates for IDL crossing...")
        lon_node = _normalize_longitude_for_idl(lon_node)

    # Extract and order boundary edges
    print("  Extracting boundary edges...")
    boundary_edges = _extract_boundary_edges(nv)
    print(f"  Found {len(boundary_edges)} boundary edges")

    print("  Ordering boundary edges...")
    ordered_nodes = _order_boundary_edges(boundary_edges)
    print(f"  Ordered {len(ordered_nodes)} boundary nodes")

    # Create boundary coordinates
    boundary_coords = [(float(lon_node[i]), float(lat_node[i])) for i in ordered_nodes]

    # Close the polygon
    if boundary_coords and boundary_coords[0] != boundary_coords[-1]:
        boundary_coords.append(boundary_coords[0])

    # Compute bounds from boundary
    lons = [c[0] for c in boundary_coords]
    lats = [c[1] for c in boundary_coords]

    lat_min = min(lats)
    lat_max = max(lats)
    lon_min = min(lons)
    lon_max = max(lons)

    # Create WKT POLYGON
    coord_str = ", ".join([f"{lon} {lat}" for lon, lat in boundary_coords])
    wkt_polygon = f"POLYGON (({coord_str}))"

    # Build result dictionary
    bounds = {
        "geospatial_bounds": wkt_polygon,
        "geospatial_bounds_crs": bounds_crs,
        "geospatial_lat_max": lat_max,
        "geospatial_lat_min": lat_min,
        "geospatial_lat_units": str(ds.lat_node.units),
        "geospatial_lat_resolution": None,  # Unstructured grid
        "geospatial_lon_max": lon_max,
        "geospatial_lon_min": lon_min,
        "geospatial_lon_units": str(ds.lon_node.units),
        "geospatial_lon_resolution": None,  # Unstructured grid
        "boundary_coordinates": boundary_coords,
    }

    # Cache result
    print(f"  Caching mesh boundary to {cache_path}")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(bounds, f, indent=2)

    # Also save as GeoJSON for visualization/verification
    geojson_path = cache_path.with_suffix(".geojson")
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "name": "mesh_exterior_boundary",
                    "crs": bounds_crs,
                    "lat_min": lat_min,
                    "lat_max": lat_max,
                    "lon_min": lon_min,
                    "lon_max": lon_max,
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [boundary_coords],
                },
            }
        ],
    }
    print(f"  Saving GeoJSON for verification to {geojson_path}")
    with open(geojson_path, "w") as f:
        json.dump(geojson, f, indent=2)

    return bounds


def _geodesic_distance(lon1, lat1, lon2, lat2):
    """
    Calculate geodesic distance between two points using WGS84 ellipsoid (vectorized).

    Uses pyproj.Geod which correctly handles the International Date Line (IDL)
    by computing actual geodesic distances on the ellipsoid.

    Parameters
    ----------
    lon1, lat1, lon2, lat2 : np.ndarray
        Coordinates in degrees

    Returns
    -------
    np.ndarray
        Distances in meters

    Raises
    ------
    ValueError
        If any calculated distances are negative (indicates a computation error)
    """
    # pyproj.Geod.inv returns (forward_azimuth, back_azimuth, distance)
    _, _, distances = _geod.inv(lon1, lat1, lon2, lat2)

    # Check for negative distances which would indicate an error
    if np.any(distances < 0):
        neg_count = np.sum(distances < 0)
        raise ValueError(
            f"Geodesic distance calculation returned {neg_count} negative values. "
            "This indicates a computation error that needs investigation."
        )

    return distances


def compute_grid_resolution(ds, config, location):
    """
    Compute grid resolution (average edge length) for each mesh element.

    Calculates the average of the three edge lengths for each triangular
    element using geodesic distance on the WGS84 ellipsoid via pyproj.

    Results are cached to avoid recomputation.

    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset containing:
        - lat_node, lon_node: node coordinates
        - nv: node connectivity (3 vertices per element)
    config : dict
        Configuration dictionary
    location : str or dict
        Location key or location configuration dict

    Returns
    -------
    dict
        Dictionary containing:
        - grid_resolution_per_face: np.ndarray of per-element resolution in meters
        - grid_resolution_mean: mean resolution across all elements
        - grid_resolution_min: minimum resolution
        - grid_resolution_max: maximum resolution
        - grid_resolution_median: median resolution
        - grid_resolution_std: standard deviation
        - grid_resolution_percentiles: dict with 5th, 25th, 75th, 95th percentiles
    """
    # Check cache first
    cache_path = _get_grid_resolution_cache_path(config, location)
    if cache_path.exists():
        print(f"Loading cached grid resolution from {cache_path}")
        with open(cache_path, "r") as f:
            cached = json.load(f)
        # Convert list back to numpy array
        cached["grid_resolution_per_face"] = np.array(
            cached["grid_resolution_per_face"]
        )
        return cached

    print("Computing grid resolution...")

    # Get coordinates and connectivity
    lat_node = ds.lat_node.values
    lon_node = ds.lon_node.values
    nv = ds.nv.values

    # Ensure shape is (n_elements, 3)
    if nv.shape[0] == 3 and nv.shape[1] != 3:
        nv = nv.T

    # Convert to 0-based indexing
    nv = nv - 1

    # Get corner coordinates for each element (in degrees for pyproj)
    lon1 = lon_node[nv[:, 0]]
    lat1 = lat_node[nv[:, 0]]
    lon2 = lon_node[nv[:, 1]]
    lat2 = lat_node[nv[:, 1]]
    lon3 = lon_node[nv[:, 2]]
    lat3 = lat_node[nv[:, 2]]

    # Calculate three edge lengths using geodesic distance
    # pyproj handles IDL crossing correctly
    edge1 = _geodesic_distance(lon1, lat1, lon2, lat2)
    edge2 = _geodesic_distance(lon2, lat2, lon3, lat3)
    edge3 = _geodesic_distance(lon3, lat3, lon1, lat1)

    # Grid resolution as average edge length
    grid_resolution = (edge1 + edge2 + edge3) / 3.0

    # Compute summary statistics
    result = {
        "grid_resolution_per_face": grid_resolution,
        "grid_resolution_mean": float(np.mean(grid_resolution)),
        "grid_resolution_min": float(np.min(grid_resolution)),
        "grid_resolution_max": float(np.max(grid_resolution)),
        "grid_resolution_median": float(np.median(grid_resolution)),
        "grid_resolution_std": float(np.std(grid_resolution)),
        "grid_resolution_percentiles": {
            "5th": float(np.percentile(grid_resolution, 5)),
            "25th": float(np.percentile(grid_resolution, 25)),
            "75th": float(np.percentile(grid_resolution, 75)),
            "95th": float(np.percentile(grid_resolution, 95)),
        },
    }

    print(f"  Mean resolution: {result['grid_resolution_mean']:.2f} m")
    print(
        f"  Min/Max: {result['grid_resolution_min']:.2f} / {result['grid_resolution_max']:.2f} m"
    )

    # Cache result (convert numpy array to list for JSON serialization)
    print(f"  Caching grid resolution to {cache_path}")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_data = result.copy()
    cache_data["grid_resolution_per_face"] = grid_resolution.tolist()
    with open(cache_path, "w") as f:
        json.dump(cache_data, f, indent=2)

    return result
