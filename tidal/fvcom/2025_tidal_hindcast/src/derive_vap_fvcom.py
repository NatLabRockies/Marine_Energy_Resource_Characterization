import gc
import multiprocessing as mp

from pathlib import Path

import numpy as np
import xarray as xr

from . import attrs_manager, file_manager, file_name_convention_manager, nc_manager

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
    "volume_energy_flux": "vap_element_volume_energy_flux",
    # "water_column_volume_average_energy_flux": "vap_water_column_mean_energy_flux",
    "zeta_center": "vap_zeta_center",
    "depth": "vap_sigma_depth",
    "sea_floor_depth": "vap_sea_floor_depth",
    "mean": "vap_water_column_mean",
    "median": "vap_water_column_median",
    "max": "vap_water_column_max",
    "p95": "vap_water_column_<PERCENTILE>th_percentile",
}


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


def calculate_sea_water_to_direction(
    ds, config, direction_undefined_speed_threshold_ms=0.0
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
        Speed threshold below which direction is set to NaN, default 0.0 m/s

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
    cartesian_angle_degrees = np.rad2deg(np.arctan2(ds.v, ds.u))

    # Validate direction ranges
    cartesian_angle_degrees_expected_max = 180.0  # arctan2 range is [-180, 180]
    cartesian_angle_degrees_expected_min = -180.0

    cartesian_angle_degrees_max = np.max(cartesian_angle_degrees)
    cartesian_angle_degrees_min = np.min(cartesian_angle_degrees)

    if cartesian_angle_degrees_max > cartesian_angle_degrees_expected_max:
        raise ValueError(
            f"Maximum mathematical direction value {cartesian_angle_degrees_max}° "
            f"exceeds expected maximum of {cartesian_angle_degrees_expected_max}°"
        )

    if cartesian_angle_degrees_min < cartesian_angle_degrees_expected_min:
        raise ValueError(
            f"Minimum mathematical direction value {cartesian_angle_degrees_min}° "
            f"is below expected minimum of {cartesian_angle_degrees_expected_min}°"
        )

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


def calculate_sea_water_from_direction(
    ds, config, direction_undefined_speed_threshold_ms=0.0
):
    """
    Calculate the direction sea water is coming FROM in meteorological convention.

    For a velocity vector with u=1, v=0 (flowing eastward):
    - from_direction = 270° (water coming from the west)

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset with 'u', 'v', and 'speed' variables, optionally 'to_direction'
    config : dict
        Configuration dictionary
    direction_undefined_speed_threshold_ms : float, optional
        Speed threshold below which direction is set to NaN, default 0.0 m/s

    Returns
    -------
    xarray.Dataset
        Input dataset with added 'from_direction' variable
    """
    # Check if to_direction already exists, if not calculate it
    if output_names["to_direction"] not in ds.variables:
        ds = calculate_sea_water_to_direction(
            ds, config, direction_undefined_speed_threshold_ms
        )

    # Convert compass 'to' direction to 'from' direction by adding 180°
    # Example: u=1, v=0 (east) has to_direction 90°
    # 90 + 180 = 270° (coming from west)
    compass_from_direction_degrees = np.mod(ds[output_names["to_direction"]] + 180, 360)

    # Set directions to NaN where speed is below threshold
    if output_names["speed"] in ds.variables:
        compass_from_direction_degrees = xr.where(
            ds[output_names["speed"]] > direction_undefined_speed_threshold_ms,
            compass_from_direction_degrees,
            np.nan,
        )

    output_variable_name = output_names["from_direction"]

    ds[output_variable_name] = compass_from_direction_degrees

    specified_attrs = config["derived_vap_specification"]["from_direction"][
        "attributes"
    ]

    # Add CF-compliant metadata
    ds[output_variable_name].attrs = {
        **specified_attrs,
        "direction_reference": (
            "Reference table for velocity components and resulting from_direction:\n"
            "| Eastward (u) | Northward (v) | From Direction | Cardinal Direction |\n"
            "| 1            | 0             | 270            | West               |\n"
            "| 1            | 1             | 225            | Southwest          |\n"
            "| 0            | 1             | 180            | South              |\n"
            "| -1           | 1             | 135            | Southeast          |\n"
            "| -1           | 0             | 90             | East               |\n"
            "| -1           | -1            | 45             | Northeast          |\n"
            "| 0            | -1            | 0              | North              |\n"
            "| 1            | -1            | 315            | Northwest          |\n"
            "| 0            | 0             | undefined      | undefined          |"
        ),
        "additional_processing": (
            f"Directions set to NaN for speeds below {direction_undefined_speed_threshold_ms} m/s."
        ),
        "computation": (
            "compass_from_direction_degrees = np.mod(ds.to_direction + 180, 360)\n"
        ),
        "input_variables": (
            "to_direction: sea_water_velocity_to_direction (degrees), "
            "speed: sea_water_velocity (m/s)"
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


def calculate_element_areas(ds):
    """
    Calculate the areas of triangular elements in an FVCOM mesh

    Parameters
    ----------
    ds : xarray.Dataset
        FVCOM dataset containing mesh information

    Returns
    -------
    numpy.ndarray
        Array of element areas in square meters
    """
    # Get the node indices for each face - first time index only
    nv = ds.nv.values[0, :, :] - 1  # Convert to 0-based indexing

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
    element_areas = 0.5 * np.sqrt(crossx**2 + crossy**2 + crossz**2)

    return element_areas


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
    element_areas = calculate_element_areas(ds)

    # Get dimensions
    n_time = len(ds.time)
    n_sigma_layer = len(ds.sigma_layer)
    n_face = len(ds.face)

    # Get bathymetry and sea surface height
    h_center = ds[output_names["h_center"]].values  # Bathymetry at element centers
    zeta_center = ds[
        output_names["zeta_center"]
    ].values  # Surface elevation at element centers

    # Get node indices for each element (face) - first time index only
    nv = ds.nv.values[0, :, :] - 1  # Convert to 0-based indexing

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
    element_areas_broadcast = element_areas.reshape(1, 1, n_face).repeat(
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


def calculate_volume_energy_flux(ds):
    """
    Calculate energy flux in each element volume.

    Parameters
    ----------
    ds : xarray.Dataset
        FVCOM dataset containing 'speed' and 'element_volume'
    config : dict, optional
        Configuration dictionary

    Returns
    -------
    xarray.Dataset
        Original dataset with added 'volume_energy_flux' variable
    """
    if output_names["power_density"] not in ds:
        raise KeyError(
            f"Dataset must contain '{output_names['power_density']}'. Please calculate sea water power_density first."
        )

    if output_names["element_volume"] not in ds:
        raise KeyError(
            f"Dataset must contain '{output_names['element_volume']}'. Please calculate element volumes first."
        )

    # Energy flux = power density * volume
    volume_energy_flux = (
        ds[output_names["power_density"]] * ds[output_names["element_volume"]]
    )

    # Add volume energy flux to dataset
    ds[output_names["volume_energy_flux"]] = volume_energy_flux

    # Add metadata
    ds[output_names["volume_energy_flux"]].attrs = {
        "long_name": "Model Element Volume Energy Flux",
        "units": "W",
        "description": "Energy flux in each element volume",
        "methodology": "Calculated as power density multiplied by element volume",
        "computation": "volume_energy_flux = power_density * element_volume",
        "input_variables": "power_density: sea water power density (W/m^2), element_volume: volume of water per element (m^3)",
    }

    return ds


# def calculate_vertical_avg_energy_flux(ds, config=None):
#     """
#     Calculate vertically averaged energy flux.
#
#     Parameters
#     ----------
#     ds : xarray.Dataset
#         FVCOM dataset containing 'volume_energy_flux' and 'element_volume'
#     config : dict, optional
#         Configuration dictionary
#
#     Returns
#     -------
#     xarray.Dataset
#         Original dataset with added 'vertical_avg_energy_flux' variable
#     """
#     if "volume_energy_flux" not in ds:
#         raise KeyError(
#             "Dataset must contain 'volume_energy_flux'. Please calculate volume energy flux first."
#         )
#
#     if "element_volume" not in ds:
#         raise KeyError(
#             "Dataset must contain 'element_volume'. Please calculate element volumes first."
#         )
#
#     # Sum energy flux and volume over all sigma layers
#     total_energy_flux = ds.volume_energy_flux.sum(dim="sigma_layer")
#     total_volume = ds.element_volume.sum(dim="sigma_layer")
#
#     # Calculate vertical average energy flux
#     vertical_avg_energy_flux = total_energy_flux / total_volume
#
#     # Add vertical average energy flux to dataset
#     ds["vertical_avg_energy_flux"] = vertical_avg_energy_flux
#
#     # Add metadata
#     ds.vertical_avg_energy_flux.attrs = {
#         "long_name": "Vertical Average Energy Flux",
#         "units": "W/m^3",
#         "description": "Energy flux averaged over the water column",
#         "methodology": "Calculated as sum of energy flux over all sigma layers divided by sum of volumes",
#         "computation": 'vertical_avg_energy_flux = sum(volume_energy_flux, dim="sigma_layer") / sum(element_volume, dim="sigma_layer")',
#         "input_variables": "volume_energy_flux: energy flux in each element volume (W), element_volume: volume of water per element (m^3)",
#     }
#
#     return ds


# def calculate_column_volume_avg_energy_flux(ds):
#     """
#     Calculate volume-weighted average energy flux for each vertical column (face).
#
#     This function computes the volume-weighted average of energy flux across the entire
#     water column for each face, providing a single value per face that better handles
#     outliers compared to simple averages of speed.
#
#     Parameters
#     ----------
#     ds : xarray.Dataset
#         FVCOM dataset containing 'volume_energy_flux' and 'element_volume'
#     config : dict, optional
#         Configuration dictionary
#
#     Returns
#     -------
#     xarray.Dataset
#         Original dataset with added 'column_volume_avg_energy_flux' variable
#     """
#     if "volume_energy_flux" not in ds:
#         raise KeyError(
#             "Dataset must contain 'volume_energy_flux'. Please calculate volume energy flux first."
#         )
#
#     if "element_volume" not in ds:
#         raise KeyError(
#             "Dataset must contain 'element_volume'. Please calculate element volumes first."
#         )
#
#     # Sum energy flux and volume over sigma layers for each face
#     # This gives us the total energy flux and total volume for each vertical column
#     total_energy_flux_per_column = ds.volume_energy_flux.sum(dim="sigma_layer")
#     total_volume_per_column = ds.element_volume.sum(dim="sigma_layer")
#
#     # Calculate volume-weighted average energy flux for each column
#     column_volume_avg_energy_flux = (
#         total_energy_flux_per_column / total_volume_per_column
#     )
#
#     # Add to dataset
#     ds["column_volume_avg_energy_flux"] = column_volume_avg_energy_flux
#
#     # Add metadata
#     ds.column_volume_avg_energy_flux.attrs = {
#         "long_name": "Column Volume-Weighted Average Energy Flux",
#         "units": "W/m^3",
#         "description": "Volume-weighted average energy flux for each vertical water column",
#         "methodology": "Calculated as sum of energy flux over all sigma layers divided by sum of volumes for each face",
#         "computation": 'column_volume_avg_energy_flux = sum(volume_energy_flux, dim="sigma_layer") / sum(element_volume, dim="sigma_layer")',
#         "input_variables": "volume_energy_flux: energy flux in each element volume (W), element_volume: volume of water per element (m^3)",
#         "visualization_purpose": "Provides a spatially-distributed metric of energy flux suitable for mapping, accounting for variable water depths and reducing the impact of outliers compared to simple speed averages",
#     }
#
#     return ds


def calculate_zeta_center(ds):
    """
    Calculate sea surface elevation at cell centers from node values.
    """

    # Get raw nv values from first timestep to avoid coordinate conflicts
    nv_values = ds.nv.isel(time=0).values - 1  # shape (3, n_faces)

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
        "long_name": "Sea Surface Height at Cell Centers",
        "coordinates": "time face",
        "mesh": "cell_centered",
        "interpolation": (
            "Computed by averaging the surface elevation values from the three "
            "nodes that define each triangular cell using the node-to-cell "
            "connectivity array (nv)"
        ),
        "computation": "zeta_center = mean(zeta[nv(t=0) - 1], axis=1)",
        "input_variables": "zeta: sea_surface_height_above_geoid at nodes",
    }

    ds[output_names["zeta_center"]] = zeta_center

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


def calculate_depth_statistics(ds, variable_name):
    """
    Calculate depth statistics (mean, median, max, (max - max[i - 1]) percentile) for a given variable.
    This function computes multiple depth-based statistics in a single pass to avoid
    redundant data access, with optimized percentile calculation.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing the variable to be analyzed
    variable_name : str
        Name of variable to calculate statistics for

    Returns
    -------
    xarray.Dataset
        Dataset with added depth statistics variables
    """
    this_output_name = output_names[variable_name]
    sanitized_this_output_name = this_output_name.replace("vap_", "")

    if this_output_name not in ds:
        raise KeyError(f"Dataset must contain '{this_output_name}'")

    dim = "sigma_layer"

    # For the high value (average of max and second max), determine the actual percentile
    n_elements = ds[this_output_name].sizes[dim]
    # Position for second-highest value in a zero-indexed array is n_elements - 2
    # Second highest is at position n_elements - 1 in 1-indexed ranking
    # So its percentile is (n_elements - 1) / n_elements * 100
    # actual_percentile = (n_elements - 1) / n_elements * 100
    actual_percentile = (
        ((n_elements / n_elements) + ((n_elements - 1) / n_elements)) / 2
    ) * 100

    # Calculate all statistics in one go
    # depth_avg_name = f"{variable_name}_depth_avg"
    depth_avg_name = f"{output_names['mean']}_{sanitized_this_output_name}"
    depth_median_name = f"{output_names['median']}_{sanitized_this_output_name}"
    depth_max_name = f"{output_names['max']}_{sanitized_this_output_name}"
    depth_percentile_name = f"{output_names['p95'].replace('<PERCENTILE>', str(int(actual_percentile)))}_{sanitized_this_output_name}"
    # depth_median_name = f"{variable_name}_depth_median"
    # depth_percentile_name = (
    #     f"{variable_name}_depth_{int(actual_percentile)}th_percentile"
    # )
    # depth_max_name = f"{variable_name}_depth_max"

    # Get original variable attributes
    orig_attrs = ds[this_output_name].attrs.copy()
    orig_long_name = orig_attrs.get("long_name", this_output_name)

    # Calculate mean
    print(f"\t\tCalculating {depth_avg_name}...")
    ds[depth_avg_name] = ds[this_output_name].mean(dim=dim)

    # Calculate median
    print(f"\t\tCalculating {depth_median_name}...")
    ds[depth_median_name] = ds[this_output_name].median(dim=dim)

    print(f"\t\tCalculating {depth_percentile_name} and {depth_max_name}...")
    # Extract the variable data as a numpy array
    var_data = ds[this_output_name].values

    # Find the axis index for the sigma_layer dimension
    axis = ds[this_output_name].dims.index(dim)

    # Partition to get the two highest values (max and second max)
    # This is much faster than sorting the entire array
    k1 = var_data.shape[axis] - 1  # Index of max value (0-indexed)
    k2 = var_data.shape[axis] - 2  # Index of second max value (0-indexed)

    # Use np.partition which partially sorts the array so elements at positions
    # >= k are in their final sorted positions
    partitioned = np.partition(var_data, [k2, k1], axis=axis)

    # Extract the max value (last element in the partitioned array)
    max_values = np.take(partitioned, k1, axis=axis)

    # Extract the second max value (second-to-last element)
    second_max_values = np.take(partitioned, k2, axis=axis)

    # Calculate the average of max and second max
    percentile_data = (max_values + second_max_values) / 2.0

    # Create a new DataArray for the high value with proper dimensions
    dims_without_depth = [d for d in ds[this_output_name].dims if d != dim]

    # Store the maximum values separately
    print(f"\t\tAdding {depth_max_name}...")
    ds[depth_max_name] = (dims_without_depth, max_values)

    print(f"\t\tAdding {depth_percentile_name}...")
    ds[depth_percentile_name] = (dims_without_depth, percentile_data)

    # Set attributes for mean
    avg_attrs = orig_attrs.copy()
    avg_attrs.pop("standard_name", None)
    ds[depth_avg_name].attrs = {
        **avg_attrs,
        "long_name": f"Depth averaged {orig_long_name}",
        "statistical_computation": "Mean across sigma layers",
    }

    # Set attributes for other calculations
    ds[depth_median_name].attrs = {
        **avg_attrs,
        "long_name": f"Depth median {orig_long_name}",
        "statistical_computation": "Median across sigma layers",
    }

    ds[depth_percentile_name].attrs = {
        **avg_attrs,
        "long_name": f"Depth {int(actual_percentile)}th percentile {orig_long_name}",
        "statistical_computation": f"Average of maximum and second maximum across sigma layers (approx. {int(actual_percentile)}th percentile)",
    }

    ds[depth_max_name].attrs = {
        **avg_attrs,
        "long_name": f"Depth maximum {orig_long_name}",
        "statistical_computation": "Maximum value across sigma layers",
    }

    return ds


def process_single_file(nc_file, config, location, output_dir, file_index):
    """Process a single netCDF file and save the results."""
    try:
        print(f"Calculating vap for {nc_file}")

        # Use context manager to ensure dataset is properly closed
        with xr.open_dataset(nc_file) as this_ds:
            print(f"\t[{file_index}] Calculating speed...")
            this_ds = calculate_sea_water_speed(this_ds, config)

            print(f"\t[{file_index}] Calculating to direction...")
            this_ds = calculate_sea_water_to_direction(this_ds, config)

            print(f"\t[{file_index}] Calculating from direction...")
            this_ds = calculate_sea_water_from_direction(this_ds, config)

            print(f"\t[{file_index}] Calculating power density...")
            this_ds = calculate_sea_water_power_density(this_ds, config)

            print(f"\t[{file_index}] Calculating zeta_center...")
            this_ds = calculate_zeta_center(this_ds)

            print(f"\t[{file_index}] Calculating depth...")
            this_ds = calculate_depth(this_ds)

            print(f"\t[{file_index}] Calculating sea_floor_depth...")
            this_ds = calculate_sea_floor_depth(this_ds)

            print(f"\t[{file_index}] Calculating element volumes...")
            this_ds = calculate_element_volume(this_ds)

            print(f"\t[{file_index}] Calculating volume energy flux...")
            this_ds = calculate_volume_energy_flux(this_ds)

            # print(f"\t[{file_index}] Calculating vertical avg energy flux...")
            # this_ds = calculate_vertical_avg_energy_flux(this_ds)

            # print(f"\t[{file_index}] Calculating column avg energy flux...")
            # this_ds = calculate_column_volume_avg_energy_flux(this_ds)
            #
            print(f"\t[{file_index}] Calculating u vertical average")
            this_ds = calculate_depth_average(this_ds, "u")

            print(f"\t[{file_index}] Calculating v vertical average")
            this_ds = calculate_depth_average(this_ds, "v")

            print(f"\t[{file_index}] Calculating to_direction vertical average")
            this_ds = calculate_depth_average(this_ds, "to_direction")

            print(f"\t[{file_index}] Calculating from_direction vertical average")
            this_ds = calculate_depth_average(this_ds, "from_direction")

            print(f"\t[{file_index}] Calculating speed depth average statistics")
            this_ds = calculate_depth_statistics(this_ds, "speed")

            print(
                f"\t[{file_index}] Calculating power_density depth average statistics"
            )
            this_ds = calculate_depth_statistics(this_ds, "power_density")

            print(
                f"\t[{file_index}] Calculating volume_energy_flux depth average statistics"
            )
            this_ds = calculate_depth_statistics(this_ds, "volume_energy_flux")

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
                f"{file_index:03d}.{data_level_file_name}",
            )

            print(f"\t[{file_index}] Saving to {output_path}...")
            # this_ds.to_netcdf(output_path, encoding=config["dataset"]["encoding"])
            this_ds.to_netcdf(
                output_path,
                encoding=nc_manager.define_compression_encoding(
                    this_ds,
                    base_encoding=config["dataset"]["encoding"],
                    compression_strategy="none",
                ),
            )

        gc.collect()

        return file_index

    except Exception as e:
        print(f"Error processing file {nc_file}: {str(e)}")
        raise e
        # Return a value indicating failure
        return -file_index


def derive_vap(config, location_key, use_multiprocessing=False):
    location = config["location_specification"][location_key]
    std_partition_path = file_manager.get_standardized_partition_output_dir(
        config, location
    )
    std_partition_nc_files = sorted(list(std_partition_path.rglob("*.nc")))

    vap_output_dir = Path(file_manager.get_vap_output_dir(config, location))

    existing_vap_nc_files = sorted(list(vap_output_dir.rglob("*.nc")))

    if len(existing_vap_nc_files) >= 12:
        print(
            f"Found {len(existing_vap_nc_files)} files in {vap_output_dir}. Skipping derive vap!"
        )
        return

    # Create a list of files to be processed, with their indices
    # Filter out files that might already be processed
    files_to_process = []
    existing_indices = set(
        int(f.name.split(".")[0])
        for f in existing_vap_nc_files
        if f.name.split(".")[0].isdigit()
    )

    for count, nc_file in enumerate(std_partition_nc_files, start=1):
        if count not in existing_indices:
            files_to_process.append((nc_file, count))

    if not files_to_process:
        print("No new files to process.")
        return

    results = []

    use_multiprocessing = False

    if use_multiprocessing is True:
        # Multiprocessing approach
        # Determine a reasonable number of processes
        # Use a smaller fraction of available CPUs to avoid memory issues
        cpu_count = mp.cpu_count()
        # Using 1/4 of available CPUs or 1, whichever is larger
        num_processes = max(1, int(cpu_count / 4))
        num_processes = 2
        print(
            f"Using {num_processes} processes to process {len(files_to_process)} vap data files"
        )

        # Set a timeout for each task to avoid indefinite hanging
        timeout_per_file = 3600  # 1 hour per file, adjust as needed

        try:
            # Process files in chunks to avoid memory overflow
            chunk_size = min(
                10, len(files_to_process)
            )  # Process max 10 files at a time
            for i in range(0, len(files_to_process), chunk_size):
                chunk = files_to_process[i : i + chunk_size]
                print(
                    f"Processing chunk {i//chunk_size + 1}/{(len(files_to_process)-1)//chunk_size + 1}"
                )
                with mp.Pool(num_processes) as pool:
                    chunk_results = pool.starmap_async(
                        process_single_file,
                        [
                            (nc_file, config, location, vap_output_dir, idx)
                            for nc_file, idx in chunk
                        ],
                    ).get(timeout=timeout_per_file * len(chunk))
                    results.extend(chunk_results)
                # Force garbage collection between chunks
                gc.collect()
            print(f"Completed processing {len(results)} files with multiprocessing.")
        except mp.TimeoutError:
            print("ERROR: Processing timed out.")
        except Exception as e:
            print(f"ERROR during multiprocessing: {str(e)}")
    else:
        # Sequential processing approach
        print(f"Processing {len(files_to_process)} vap data files sequentially")

        for nc_file, idx in files_to_process:
            print(f"Processing file {idx}/{len(files_to_process)}: {nc_file}")
            result = process_single_file(nc_file, config, location, vap_output_dir, idx)
            results.append(result)

            # Force garbage collection after each file
            gc.collect()

        print(f"Completed processing {len(results)} files sequentially.")

    # Check for any errors in processing (for both approaches)
    failed_files = [i for i in results if i < 0]
    if failed_files:
        print(
            f"WARNING: {len(failed_files)} files failed to process: {[-i for i in failed_files]}"
        )
