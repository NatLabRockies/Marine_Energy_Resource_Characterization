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


def standardize_fvcom_coords(ds, utm_zone: int = None):
    coord_system = ds.attrs.get("CoordinateSystem")
    coord_projection = ds.attrs.get("CoordinateProjection", "none")
    if not coord_system:
        raise ValueError("NetCDF file missing CoordinateSystem attribute")

    original_lat_centers = ds["latc"].values
    original_lon_centers = ds["lonc"].values
    original_lat_corners = ds["lat"].values
    original_lon_corners = ds["lon"].values

    # If coordinates are zeros, use conversion
    if np.allclose(original_lat_centers, 0.0) and np.allclose(
        original_lon_centers, 0.0
    ):
        original_utm_x_centers = ds["xc"].values
        original_utm_y_centers = ds["yc"].values
        original_utm_x_corners = ds["x"].values
        original_utm_y_corners = ds["y"].values
        transformer = create_transformer(coord_system, coord_projection, utm_zone)
        if transformer:
            lon_centers, lat_centers = transformer.transform(
                original_utm_x_centers, original_utm_y_centers
            )
            lon_corners, lat_corners = transformer.transform(
                original_utm_x_corners, original_utm_y_corners
            )
        else:
            raise ValueError("Zero coordinates but no transformation defined")
    else:
        lat_centers = original_lat_centers
        lon_centers = original_lon_centers
        lat_corners = original_lat_corners
        lon_corners = original_lon_corners

    # Reorganize corners to match centers using nv mapping
    corner_indices = get_node_to_cell_mapping(ds)

    # Reorder corners to match centers
    lat_corners_mapped = lat_corners[corner_indices]  # Shape: (nele, 3)
    lon_corners_mapped = lon_corners[corner_indices]  # Shape: (nele, 3)

    if lat_corners_mapped.shape[1] != 3:
        raise ValueError(
            f"Expected exactly 3 lat corners per cell, got {lat_corners_mapped.shape[1]}"
        )

    if lon_corners_mapped.shape[1] != 3:
        raise ValueError(
            f"Expected exactly 3 lon corners per cell, got {lon_corners_mapped.shape[1]}"
        )

    # Verify centers are within corners
    def verify_centers_in_corners(centers, corners):
        # Simple bounding box check for each element
        min_bounds = np.min(corners, axis=1)  # Shape: (nele,)
        max_bounds = np.max(corners, axis=1)  # Shape: (nele,)
        within_bounds = (centers >= min_bounds) & (centers <= max_bounds)
        return np.all(within_bounds)

    centers_valid = verify_centers_in_corners(lat_centers, lat_corners_mapped)
    if not centers_valid:
        print("Warning: Some center points lie outside their corner bounds")

    return {
        "lat_centers": lat_centers,
        "lon_centers": lon_centers,
        "lat_corners": lat_corners_mapped,
        "lon_corners": lon_corners_mapped,
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
