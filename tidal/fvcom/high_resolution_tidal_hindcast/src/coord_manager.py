import pyproj
import numpy as np

# Define output CRS using explicit WGS84 geographic parameters
# https://en.wikipedia.org/wiki/World_Geodetic_System
# TODO: This is not defined specifically in any dataset, or in any documentation
# It would be wise to figure this out prior to deployment
# https://vdatum.noaa.gov/docs/datums.html
# https://www.fvcom.org/?p=15
# Proj4: It is used for coordinate conversion . FVCOM includes both Cartesian and spherical coordinates. The WRF NetCDF output uses the earth coordinate system defined as longitudes and latitudes. If the Cartesian coordinate is selected for FVCOM, Proj4 is required to turn on to convert longitudes and latitudes to the x-y coordinates. Proj4 is also used for the FVCOM output if the earth coordinates system is selected for the model run in the Cartesian coordinates.
# https://fabienmaussion.info/2018/01/06/wrf-projection/
#  use the WRF spherical earth radius parameters to define the WRF map projection, and consider the lon/lat coordinates obtained this way as being valid in WGS84.
OUTPUT_COORDINATE_REFERENCE_SYSTEM = pyproj.CRS(
    proj="latlon", datum="WGS84", ellps="WGS84"
)


def create_transformer(coord_system: str, coord_projection: str, utm_zone: int = None):
    """Create transformer based on netCDF attributes"""
    # Do not alter existing gps coordinates, aleutian_islands, cook_inlet
    if coord_system == "GeoReferenced":
        return None

    output_crs = OUTPUT_COORDINATE_REFERENCE_SYSTEM

    # For custom projections, piscatqua_river, and western_passage
    # piscataqua_river
    # CoordinateSystem	Cartesian
    # CoordinateProjection	proj=tmerc +datum=NAD83 +lon_0=-69 lat_0=0 k=.999600000 x_0=500000 y_0=0
    # western_passage
    # CoordinateSystem	Cartesian
    # CoordinateProjection	proj=tmerc +datum=NAD83 +lon_0=-70d10 lat_0=42d50 k=.9999666666666667 x_0=900000 y_0=0
    # if coord_system == "Cartesian" and coord_projection.startswith("proj="):
    #     # Handle degree-minute notation in projection strings (e.g., -70d10 → -70.16667)
    #     import re
    #
    #     def convert_degree_minute_to_decimal(match):
    #         # Extract the sign, degrees, and minutes
    #         full_match = match.group(0)
    #         sign = -1 if full_match.startswith("-") else 1
    #         degree_part = match.group(1)
    #         minute_part = match.group(2)
    #
    #         # Convert degrees and minutes to decimal degrees
    #         degrees = float(degree_part.replace("-", "").replace("+", ""))
    #         minutes = float(minute_part)
    #         decimal_degrees = sign * (degrees + minutes / 60.0)
    #
    #         return str(decimal_degrees)
    #
    #     # Use regex to find and replace degree-minute notation
    #     # This pattern matches: optional - or +, followed by digits, followed by 'd',
    #     # followed by more digits
    #     degree_minute_pattern = r"([-+]?\d+)d(\d+)"
    #     fixed_proj = re.sub(
    #         degree_minute_pattern, convert_degree_minute_to_decimal, coord_projection
    #     )
    #
    #     if fixed_proj != coord_projection:
    #         print("Converted degree-minute notation in projection string:")
    #         print(f"Original: {coord_projection}")
    #         print(f"Converted: {fixed_proj}")
    #
    #     try:
    #         source_crs = pyproj.CRS.from_proj4(fixed_proj)
    #     except Exception as e:
    #         print(f"Error parsing projection string: {e}")
    #         print("Attempting to use original projection string")
    #         source_crs = pyproj.CRS.from_proj4(coord_projection)

    # For UTM with no projection string, puget_sound
    # puget_sound
    # CoordinateSystem	Cartesian
    # CoordinateProjection	none
    # For puget_sound the projection string or UTM zone is not specified anywhere is the original data.
    # It is derived from the name of the dataset and verified by plotting points on a map
    # elif coord_system == "Cartesian" and coord_projection == "none" and utm_zone:
    if coord_system == "Cartesian" and utm_zone:
        # https://epsg.io/26910 (Puget Sound)
        # https://epsg.io/26919 (Western Passage and Piscataqua River)
        utm_epsg = f"269{str(utm_zone).zfill(2)}"  # NAD83 UTM zones, using NAD83 based on the above datasets
        source_crs = pyproj.CRS(f"EPSG:{utm_epsg}")
        print(f"Using UTM zone {utm_zone} with EPSG code {utm_epsg}")
    else:
        raise ValueError(
            f"Unsupported coordinate configuration - System: {coord_system}, Projection: {coord_projection}"
        )

    return pyproj.Transformer.from_crs(source_crs, output_crs, always_xy=True)


def get_node_to_cell_mapping(ds):
    """Get mapping from nodes to cells using nv connectivity array"""
    # nv indices are 1-based, subtract 1
    return ds["nv"].values.T - 1  # Shape: (nele, 3)


def verify_latitude_range(lat, variable_name="latitude"):
    """
    Verify that latitude values are within the valid range [-90, 90].

    Args:
        lat: Array of latitude values
        variable_name: Name of the variable for error messages

    Raises:
        ValueError: If any latitude values are outside [-90, 90] range
    """
    lat = np.asarray(lat)
    if not np.all(np.isfinite(lat)):
        raise ValueError(f"Found non-finite {variable_name} values")

    min_lat, max_lat = np.min(lat), np.max(lat)
    if min_lat < -90 or max_lat > 90:
        raise ValueError(
            f"{variable_name} values must be in range [-90, 90], "
            f"found range [{min_lat:.2f}, {max_lat:.2f}]"
        )


def verify_longitude_range(lon, variable_name="longitude"):
    """
    Verify that longitude values are within the valid range [-180, 180].

    Args:
        lon: Array of longitude values
        variable_name: Name of the variable for error messages

    Raises:
        ValueError: If any longitude values are outside [-180, 180] range
    """
    lon = np.asarray(lon)
    if not np.all(np.isfinite(lon)):
        raise ValueError(f"Found non-finite {variable_name} values")

    min_lon, max_lon = np.min(lon), np.max(lon)
    if min_lon < -180 or max_lon > 180:
        raise ValueError(
            f"{variable_name} values must be in range [-180, 180], "
            f"found range [{min_lon:.2f}, {max_lon:.2f}]"
        )


def normalize_longitude(lon):
    """
    Normalize longitude values to the range [-180, 180].
    Handles arrays of longitudes that may cross the international date line.

    Args:
        lon: Longitude values in degrees

    Returns:
        Normalized longitude values
    """
    lon = np.asarray(lon)
    return ((lon + 180) % 360) - 180


def adjust_longitudes_for_dateline(lons):
    """
    Adjust longitude values when they cross the international date line
    to ensure proper geometric calculations.

    Args:
        lons: Array of longitude values that may cross the date line

    Returns:
        Adjusted longitude values
    """
    mean_lon = np.mean(lons)
    if mean_lon > 0:
        return np.where(lons < 0, lons + 360, lons)
    else:
        return np.where(lons > 0, lons - 360, lons)


def verify_centers_in_faces(centers, face_vertices):
    """
    Verify that cell centers lie within their triangular faces,
    handling cases where triangles cross the international date line.

    Args:
        centers: List of (lon, lat) center points
        face_vertices: List of triangles, each with 3 (lon, lat) vertices

    Returns:
        bool: True if all centers are within their faces
    """

    def point_in_triangle(point, triangle):
        # Extract coordinates
        x, y = point
        x1, y1 = triangle[0]
        x2, y2 = triangle[1]
        x3, y3 = triangle[2]

        # Check if triangle crosses date line by looking for large longitude differences
        lons = np.array([x1, x2, x3, x])
        if np.max(np.abs(np.diff(lons))) > 180:
            # Adjust longitudes to handle date line crossing
            lons = adjust_longitudes_for_dateline(lons)
            x1, x2, x3, x = lons

        # Calculate barycentric coordinates
        denom = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
        if np.abs(denom) < 1e-10:  # Handle degenerate triangles
            return False

        a = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / denom
        b = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / denom
        c = 1 - a - b

        # Check if point is inside triangle with small tolerance for numerical precision
        eps = 1e-10
        return (
            (-eps <= a <= 1 + eps) and (-eps <= b <= 1 + eps) and (-eps <= c <= 1 + eps)
        )

    # Check each center against its corresponding triangular face
    for center, triangle in zip(centers, face_vertices):
        if not point_in_triangle(center, triangle):
            return False
    return True


def standardize_fvcom_coords(ds, utm_zone=None):
    """
    Standardize FVCOM coordinates, handling cases where geometries cross the international date line.

    Args:
        ds: xarray Dataset containing FVCOM grid data
        utm_zone: Optional UTM zone number for coordinate conversion

    Returns:
        dict: Standardized coordinate information
    """
    coord_system = ds.attrs.get("CoordinateSystem")
    coord_projection = ds.attrs.get("CoordinateProjection", "none")
    if not coord_system:
        raise ValueError("NetCDF file missing CoordinateSystem attribute")

    # Get original coordinates
    original_lat_centers = ds["latc"].values
    original_lon_centers = ds["lonc"].values
    original_lat_nodes = ds["lat"].values
    original_lon_nodes = ds["lon"].values

    # Handle zero coordinates case
    if np.allclose(original_lat_centers, 0.0) and np.allclose(
        original_lon_centers, 0.0
    ):
        original_utm_x_centers = ds["xc"].values
        original_utm_y_centers = ds["yc"].values
        original_utm_x_nodes = ds["x"].values
        original_utm_y_nodes = ds["y"].values
        transformer = create_transformer(coord_system, coord_projection, utm_zone)
        if transformer:
            lon_centers, lat_centers = transformer.transform(
                original_utm_x_centers, original_utm_y_centers
            )
            lon_nodes, lat_nodes = transformer.transform(
                original_utm_x_nodes, original_utm_y_nodes
            )
        else:
            raise ValueError("Zero coordinates but no transformation defined")
    else:
        lat_centers = original_lat_centers
        lon_centers = original_lon_centers
        lat_nodes = original_lat_nodes
        lon_nodes = original_lon_nodes

    # Normalize all longitudes consistently
    lon_centers = normalize_longitude(lon_centers)
    lon_nodes = normalize_longitude(lon_nodes)

    # Verify coordinate ranges
    verify_latitude_range(lat_centers)
    verify_longitude_range(lon_centers)
    verify_latitude_range(lat_nodes)
    verify_longitude_range(lon_nodes)

    # Get node connectivity and face coordinates
    face_node_indices = get_node_to_cell_mapping(ds)
    lat_face_nodes = lat_nodes[face_node_indices]
    lon_face_nodes = lon_nodes[face_node_indices]

    if lat_face_nodes.shape[1] != 3 or lon_face_nodes.shape[1] != 3:
        raise ValueError("Expected exactly 3 nodes per face")

    # Prepare data for validation
    center_points = list(zip(lon_centers, lat_centers))
    face_triangles = [
        list(zip(lons, lats)) for lons, lats in zip(lon_face_nodes, lat_face_nodes)
    ]

    # Validate centers with date line handling
    if not verify_centers_in_faces(center_points, face_triangles):
        raise ValueError("Some face centers lie outside their face bounds")

    return {
        "lat_centers": lat_centers,
        "lon_centers": lon_centers,
        "lat_nodes": lat_nodes,
        "lon_nodes": lon_nodes,
        "lat_face_nodes": lat_face_nodes,
        "lon_face_nodes": lon_face_nodes,
    }


if __name__ == "__main__":
    import xarray as xr

    ds = xr.open_dataset("../data/00_raw/PIR_0240.nc", decode_times=False)

    coords = standardize_fvcom_coords(
        ds,
        # 19,
        None,
    )

    print(coords["lat_centers"][:5])
    print(coords["lon_centers"][:5])
    print(coords["lat_corners"][:5])
    print(coords["lon_corners"][:5])
