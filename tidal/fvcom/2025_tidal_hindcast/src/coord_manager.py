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
    if coord_system == "Cartesian" and coord_projection.startswith("proj="):
        source_crs = pyproj.CRS.from_proj4(coord_projection)

    # For UTM with no projection string, puget_sound
    # puget_sound
    # CoordinateSystem	Cartesian
    # CoordinateProjection	none
    # For puget_sound the projection string or UTM zone is not specified anywhere is the original data.
    # It is derived from the name of the dataset and verified by plotting points on a map
    elif coord_system == "Cartesian" and coord_projection == "none" and utm_zone:
        # https://epsg.io/26910 (Puget Sound)
        utm_epsg = f"269{str(utm_zone).zfill(2)}"  # NAD83 UTM zones, using NAD83 based on the above datasets
        source_crs = pyproj.CRS(f"EPSG:{utm_epsg}")
    else:
        raise ValueError(
            f"Unsupported coordinate configuration - System: {coord_system}, Projection: {coord_projection}"
        )

    return pyproj.Transformer.from_crs(source_crs, output_crs, always_xy=True)


def get_node_to_cell_mapping(ds):
    """Get mapping from nodes to cells using nv connectivity array"""
    # nv indices are 1-based, subtract 1
    return ds["nv"].values.T - 1  # Shape: (nele, 3)


def normalize_longitude(lon):
    """
    Normalize longitude values to the range [-180, 180].

    Args:
        lon: Longitude values in degrees

    Returns:
        Normalized longitude values
    """
    return np.where(lon > 180, lon - 360, lon)


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


def standardize_fvcom_coords(ds, utm_zone: int = None):
    coord_system = ds.attrs.get("CoordinateSystem")
    coord_projection = ds.attrs.get("CoordinateProjection", "none")
    if not coord_system:
        raise ValueError("NetCDF file missing CoordinateSystem attribute")

    # Get original coordinates for face centers and nodes
    original_lat_centers = ds["latc"].values
    original_lon_centers = ds["lonc"].values
    original_lat_nodes = ds["lat"].values
    original_lon_nodes = ds["lon"].values

    # If coordinates are zeros, use conversion
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

    # Normalize and verify all coordinates
    lon_centers = normalize_longitude(lon_centers)
    lon_nodes = normalize_longitude(lon_nodes)

    verify_latitude_range(lat_centers)
    verify_longitude_range(lon_centers)
    verify_latitude_range(lat_nodes)
    verify_longitude_range(lon_nodes)

    # Get the node connectivity for each face
    face_node_indices = get_node_to_cell_mapping(ds)

    # Get coordinates of nodes for each face using connectivity
    lat_face_nodes = lat_nodes[face_node_indices]  # Shape: (nfaces, 3)
    lon_face_nodes = lon_nodes[face_node_indices]  # Shape: (nfaces, 3)

    if lat_face_nodes.shape[1] != 3:
        raise ValueError(
            f"Expected exactly 3 nodes per face, got {lat_face_nodes.shape[1]}"
        )
    if lon_face_nodes.shape[1] != 3:
        raise ValueError(
            f"Expected exactly 3 nodes per face, got {lon_face_nodes.shape[1]}"
        )

    # Verify centers are within triangular faces
    def verify_centers_in_faces(centers, face_vertices):
        def point_in_triangle(point, triangle):
            # Implementation using barycentric coordinates
            x, y = point
            x1, y1 = triangle[0]
            x2, y2 = triangle[1]
            x3, y3 = triangle[2]
            # Calculate barycentric coordinates
            denom = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
            a = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / denom
            b = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / denom
            c = 1 - a - b
            # Check if point is inside triangle
            return (0 <= a <= 1) and (0 <= b <= 1) and (0 <= c <= 1)

        # Check each center against its corresponding triangular face
        for center, triangle in zip(centers, face_vertices):
            if not point_in_triangle(center, triangle):
                return False
        return True

    center_points = list(zip(lon_centers, lat_centers))
    face_triangles = [
        list(zip(lons, lats)) for lons, lats in zip(lon_face_nodes, lat_face_nodes)
    ]
    centers_valid = verify_centers_in_faces(center_points, face_triangles)

    if not centers_valid:
        raise ValueError("Warning: Some face centers lie outside their face bounds")

    return {
        "lat_centers": lat_centers,
        "lon_centers": lon_centers,
        "lat_nodes": lat_nodes,  # Original node coordinates
        "lon_nodes": lon_nodes,  # Original node coordinates
        "lat_face_nodes": lat_face_nodes,  # Node coordinates per face (nfaces, 3)
        "lon_face_nodes": lon_face_nodes,  # Node coordinates per face (nfaces, 3)
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
