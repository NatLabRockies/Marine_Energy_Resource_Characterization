import gc

from pathlib import Path

import numpy as np
import xarray as xr

from . import attrs_manager, file_manager, file_name_convention_manager


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

    output_variable_name = "speed"

    # Calculate speed maintaining original dimensions
    ds[output_variable_name] = np.sqrt(ds.u**2 + ds.v**2)

    specified_attrs = config["derived_vap_specification"][output_variable_name][
        "attributes"
    ]

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


def calculate_sea_water_direction(
    ds, config, direction_undefined_speed_threshold_ms=0.0
):
    """
    Calculate sea water velocity direction in meteorological convention.

    This function computes the direction water is coming FROM (like wind direction),
    measured clockwise from true north.

    The conversion from cartesian to meteorological 'from' direction works in two steps:
    1. Convert cartesian angle to compass 'to' direction:
       - Subtract cartesian angle from 90° to change reference from east to north
       - Use modulo 360 to normalize to [0, 360) range

    2. Convert compass 'to' direction to 'from' direction:
       - Add 180° to get the opposite direction
       - Use modulo 360 to normalize to [0, 360) range

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing:
        - 'u': eastward sea water velocity component
        - 'v': northward sea water velocity component
        - 'speed': magnitude of velocity (must be pre-computed)
    direction_undefined_speed_threshold_ms : float, optional
        Speed threshold below which direction is set to NaN since direction
        becomes meaningless for near-zero velocities. Default is 0.0 m/s.

    Returns
    -------
    xarray.Dataset
        Original dataset with added 'sea_water_velocity_from_direction' variable

    Notes
    -----
    Direction values follow these conventions:
    - 0° : water flowing from north to south
    - 90° : water flowing from east to west
    - 180° : water flowing from south to north
    - 270° : water flowing from west to east

    """
    validate_u_and_v(ds)

    if "speed" not in ds.variables:
        raise KeyError(
            "Dataset must contain 'speed'. "
            "Please run calculate_sea_water_speed() first."
        )

    # Calculate cartesian angle using arctan2 (counterclockwise from east)
    # arctan2 radian angle is measured counterclockwise from the positive x-axis
    # numpy outputs from 0 to 180 on the positive y and from 0 to -180 on the negative y
    # Compass 270 degrees is 180 in numpy
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

    # Convert to compass to direction:
    # | u, v  | Cartesian Angle | 90 - Cartesian Angle | Modulo 360 (to_direction) | Compass To | (to + 180) % 360 (from_direction) | Compass From
    # | 1, 0  | 0               | 90 - 0 = 90          | 90 % 360 = 90             | E          | (90 + 180) % 360 = 270            | W
    # | 1, 1  | 45              | 90 - 45 = 45         | 45 % 360 = 45             | NE         | (45 + 180) % 360 = 225            | SW
    # | 0, 1  | 90              | 90 - 90 = 0          | 0 % 360 = 0               | N          | (0 + 180) % 360 = 180             | S
    # | -1, 1 | 135             | 90 - 135 = -45       | -45 % 360 = 315           | NW         | (315 + 180) % 360 = 135           | SE
    # | -1, 0 | 180             | 90 - 180 = -90       | -90 % 360 = 270           | W          | (270 + 180) % 360 = 90            | E
    # | -1,-1 | -135            | 90 - -135 = 225      | 225 % 360 = 225           | SW         | (225 + 180) % 360 = 45            | NE
    # | 0, -1 | -90             | 90 - -90 = 180       | 180 % 360 = 180           | S          | (180 + 180) % 360 = 0             | N
    # | 1, -1 | -45             | 90 - -45 = 135       | 135 % 360 = 135           | SE         | (135 + 180) % 360 = 315           | NW
    compass_to_direction_degrees = np.mod(90 - cartesian_angle_degrees, 360)
    compass_from_direction_degrees = np.mod(compass_to_direction_degrees + 180, 360)

    compass_direction_degrees_expected_max = 360
    compass_direction_degrees_expected_min = 0

    compass_direction_degrees_max = np.max(compass_from_direction_degrees)
    compass_direction_degrees_min = np.min(compass_from_direction_degrees)

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
    compass_from_direction_degrees = xr.where(
        ds.speed > direction_undefined_speed_threshold_ms,
        compass_from_direction_degrees,
        np.nan,
    )

    output_variable_name = "from_direction"

    ds[output_variable_name] = compass_from_direction_degrees

    specified_attrs = config["derived_vap_specification"][output_variable_name][
        "attributes"
    ]

    # Add CF-compliant metadata
    ds.from_direction.attrs = {
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
            "Direction is calculated using numpy.arctan2(v,u) to get the cartesian angle, "
            "converting to compass 'to' direction by (90° - cartesian angle) mod 360, "
            "then converting to compass 'from' direction by (compass_to_direction + 180°) mod 360. "
            f"Directions are set to NaN for speeds below {direction_undefined_speed_threshold_ms} m/s."
        ),
        "computation": (
            "cartesian_direction_degrees = np.rad2deg(np.arctan2(ds.v, ds.u))\n"
            "compass_to_direction_degrees = np.mod(90 - cartesian_degrees, 360)\n"
            "compass_from_direction_degrees = np.mod(compass_to_direction_degrees + 180, 360)\n"
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

    if "speed" not in ds:
        raise KeyError(
            "Dataset must contain 'speed'. "
            "Please run calculate_sea_water_speed() first."
        )

    output_variable_name = "power_density"

    # Calculate power density using Equation 1 from
    # Haas, Kevin A., et al. "Assessment of Energy Production Potential from
    # Tidal Streams in the United States." , Jun. 2011. https://doi.org/10.2172/1219367
    ds[output_variable_name] = 0.5 * rho * ds.speed**3

    specified_attrs = config["derived_vap_specification"][output_variable_name][
        "attributes"
    ]

    # Add CF-compliant metadata
    ds.power_density.attrs = {
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
    Calculate the areas of triangular elements in an FVCOM mesh.

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
    # Get single time nv, this should not change
    nv = ds.nv.values[0, :, :]
    nv = nv - 1  # Convert to 0-based indexing if needed

    # Get node coordinates
    lon_node = ds.lon_node.values
    lat_node = ds.lat_node.values

    # Constants for Earth calculations
    R_EARTH = 6371000  # Earth radius in meters
    DEG_TO_RAD = np.pi / 180

    # Initialize the areas array
    n_face = len(ds.face)
    element_areas = np.zeros(n_face)

    for i in range(n_face):
        # Get the three nodes of this element
        n1, n2, n3 = nv[:, i]

        # Convert to radians
        lat1, lon1 = lat_node[n1] * DEG_TO_RAD, lon_node[n1] * DEG_TO_RAD
        lat2, lon2 = lat_node[n2] * DEG_TO_RAD, lon_node[n2] * DEG_TO_RAD
        lat3, lon3 = lat_node[n3] * DEG_TO_RAD, lon_node[n3] * DEG_TO_RAD

        # Convert to cartesian coordinates
        x1 = R_EARTH * np.cos(lat1) * np.cos(lon1)
        y1 = R_EARTH * np.cos(lat1) * np.sin(lon1)
        z1 = R_EARTH * np.sin(lat1)

        x2 = R_EARTH * np.cos(lat2) * np.cos(lon2)
        y2 = R_EARTH * np.cos(lat2) * np.sin(lon2)
        z2 = R_EARTH * np.sin(lat2)

        x3 = R_EARTH * np.cos(lat3) * np.cos(lon3)
        y3 = R_EARTH * np.cos(lat3) * np.sin(lon3)
        z3 = R_EARTH * np.sin(lat3)

        # Vectors for two sides of the triangle
        v1 = np.array([x2 - x1, y2 - y1, z2 - z1])
        v2 = np.array([x3 - x1, y3 - y1, z3 - z1])

        # Cross product for area
        cross = np.cross(v1, v2)
        area = 0.5 * np.sqrt(np.sum(cross**2))
        element_areas[i] = area

    return element_areas


def calculate_element_volumes(ds):
    """
    Calculate volumes for each element at each time step and sigma layer using
    actual sigma layer and level information from the dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        FVCOM dataset containing bathymetry, surface elevation, and mesh information
    config : dict, optional
        Configuration dictionary

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
    n_node = len(ds.node)

    # Initialize volume array
    element_volumes = np.zeros((n_time, n_sigma_layer, n_face))

    # Get bathymetry and sea surface height
    h_center = ds.h_center.values  # Bathymetry at element centers
    zeta_center = ds.zeta_center.values  # Surface elevation at element centers

    # Get node indices for each element (face)
    nv = ds.nv.values - 1  # Convert to 0-based indexing if needed

    # Get sigma layer and level values
    # In FVCOM, sigma_layer represents mid-points and sigma_level represents interfaces
    sigma_layer_values = ds.sigma_layer.values  # Shape: (n_sigma_layer, n_node)
    sigma_level_values = ds.sigma_level.values  # Shape: (n_sigma_level, n_node)

    # Calculate layer thicknesses at nodes
    # Layer thickness is the difference between adjacent sigma levels
    layer_thickness_at_nodes = np.zeros((n_sigma_layer, n_node))
    for layer in range(n_sigma_layer):
        # Note: sigma coordinates are negative (0 at surface, -1 at bottom)
        # so the thickness is the absolute difference
        layer_thickness_at_nodes[layer, :] = np.abs(
            sigma_level_values[layer + 1, :] - sigma_level_values[layer, :]
        )

    # Calculate element volumes for each time step and layer
    for t in range(n_time):
        # Total water depth at this time step (bathymetry + surface elevation)
        total_depth = np.abs(h_center[t]) + zeta_center[t]

        for layer in range(n_sigma_layer):
            for face in range(n_face):
                # Get nodes for this element
                node1, node2, node3 = nv[:, face]

                # Get layer thickness at each node of this element
                thickness1 = layer_thickness_at_nodes[layer, node1]
                thickness2 = layer_thickness_at_nodes[layer, node2]
                thickness3 = layer_thickness_at_nodes[layer, node3]

                # Average layer thickness for this element
                avg_layer_thickness = (thickness1 + thickness2 + thickness3) / 3.0

                # Layer thickness as a fraction of total water depth
                layer_thickness_meters = total_depth[face] * avg_layer_thickness

                # Volume = area * thickness
                element_volumes[t, layer, face] = (
                    element_areas[face] * layer_thickness_meters
                )

    # Add element volumes to dataset
    ds["element_volume"] = xr.DataArray(
        element_volumes,
        dims=["time", "sigma_layer", "face"],
        attrs={
            "long_name": "Element Volume",
            "standard_name": "volume_of_water_per_element",
            "units": "m^3",
            "description": "Volume of each triangular element at each time step and sigma layer",
            "methodology": "Calculated as element area multiplied by layer thickness",
            "computation": "element_volume = element_area * (total_water_depth * sigma_layer_thickness)",
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
    if "speed" not in ds:
        raise KeyError(
            "Dataset must contain 'speed'. Please calculate sea water speed first."
        )

    if "element_volume" not in ds:
        raise KeyError(
            "Dataset must contain 'element_volume'. Please calculate element volumes first."
        )

    # Energy flux = power density * volume
    # Power density is already calculated as 0.5 * rho * speed^3
    volume_energy_flux = ds.power_density * ds.element_volume

    # Add volume energy flux to dataset
    ds["volume_energy_flux"] = volume_energy_flux

    # Add metadata
    ds.volume_energy_flux.attrs = {
        "long_name": "Volume Energy Flux",
        "units": "W",
        "description": "Energy flux in each element volume",
        "methodology": "Calculated as power density multiplied by element volume",
        "computation": "volume_energy_flux = power_density * element_volume",
        "input_variables": "power_density: sea water power density (W/m^2), element_volume: volume of water per element (m^3)",
    }

    return ds


def calculate_vertical_avg_energy_flux(ds, config=None):
    """
    Calculate vertically averaged energy flux.

    Parameters
    ----------
    ds : xarray.Dataset
        FVCOM dataset containing 'volume_energy_flux' and 'element_volume'
    config : dict, optional
        Configuration dictionary

    Returns
    -------
    xarray.Dataset
        Original dataset with added 'vertical_avg_energy_flux' variable
    """
    if "volume_energy_flux" not in ds:
        raise KeyError(
            "Dataset must contain 'volume_energy_flux'. Please calculate volume energy flux first."
        )

    if "element_volume" not in ds:
        raise KeyError(
            "Dataset must contain 'element_volume'. Please calculate element volumes first."
        )

    # Sum energy flux and volume over all sigma layers
    total_energy_flux = ds.volume_energy_flux.sum(dim="sigma_layer")
    total_volume = ds.element_volume.sum(dim="sigma_layer")

    # Calculate vertical average energy flux
    vertical_avg_energy_flux = total_energy_flux / total_volume

    # Add vertical average energy flux to dataset
    ds["vertical_avg_energy_flux"] = vertical_avg_energy_flux

    # Add metadata
    ds.vertical_avg_energy_flux.attrs = {
        "long_name": "Vertical Average Energy Flux",
        "units": "W/m^3",
        "description": "Energy flux averaged over the water column",
        "methodology": "Calculated as sum of energy flux over all sigma layers divided by sum of volumes",
        "computation": 'vertical_avg_energy_flux = sum(volume_energy_flux, dim="sigma_layer") / sum(element_volume, dim="sigma_layer")',
        "input_variables": "volume_energy_flux: energy flux in each element volume (W), element_volume: volume of water per element (m^3)",
    }

    return ds


def calculate_volume_avg_energy_flux(ds, config=None):
    """
    Calculate volume-weighted average energy flux across the entire domain.

    This function computes the total energy flux in the entire domain divided by
    the total volume, providing a single value for each time step that represents
    the volume-weighted average energy flux across all elements.

    Parameters
    ----------
    ds : xarray.Dataset
        FVCOM dataset containing 'volume_energy_flux' and 'element_volume'
    config : dict, optional
        Configuration dictionary

    Returns
    -------
    xarray.Dataset
        Original dataset with added 'volume_avg_energy_flux' variable
    """
    if "volume_energy_flux" not in ds:
        raise KeyError(
            "Dataset must contain 'volume_energy_flux'. Please calculate volume energy flux first."
        )

    if "element_volume" not in ds:
        raise KeyError(
            "Dataset must contain 'element_volume'. Please calculate element volumes first."
        )

    # Sum energy flux and volume over all elements and sigma layers
    total_energy_flux = ds.volume_energy_flux.sum(dim=["sigma_layer", "face"])
    total_volume = ds.element_volume.sum(dim=["sigma_layer", "face"])

    # Calculate volume average energy flux
    volume_avg_energy_flux = total_energy_flux / total_volume

    # Add volume average energy flux to dataset
    ds["volume_avg_energy_flux"] = volume_avg_energy_flux

    # Add metadata
    ds.volume_avg_energy_flux.attrs = {
        "long_name": "Volume Average Energy Flux",
        "units": "W/m^3",
        "description": "Energy flux averaged over the entire domain volume",
        "methodology": "Calculated as sum of energy flux over all elements and sigma layers divided by total volume",
        "computation": 'volume_avg_energy_flux = sum(volume_energy_flux, dim=["sigma_layer", "face"]) / sum(element_volume, dim=["sigma_layer", "face"])',
        "input_variables": "volume_energy_flux: energy flux in each element volume (W), element_volume: volume of water per element (m^3)",
    }

    return ds


def calculate_column_volume_avg_energy_flux(ds, config=None):
    """
    Calculate volume-weighted average energy flux for each vertical column (face).

    This function computes the volume-weighted average of energy flux across the entire
    water column for each face, providing a single value per face that better handles
    outliers compared to simple averages of speed.

    Parameters
    ----------
    ds : xarray.Dataset
        FVCOM dataset containing 'volume_energy_flux' and 'element_volume'
    config : dict, optional
        Configuration dictionary

    Returns
    -------
    xarray.Dataset
        Original dataset with added 'column_volume_avg_energy_flux' variable
    """
    if "volume_energy_flux" not in ds:
        raise KeyError(
            "Dataset must contain 'volume_energy_flux'. Please calculate volume energy flux first."
        )

    if "element_volume" not in ds:
        raise KeyError(
            "Dataset must contain 'element_volume'. Please calculate element volumes first."
        )

    # Sum energy flux and volume over sigma layers for each face
    # This gives us the total energy flux and total volume for each vertical column
    total_energy_flux_per_column = ds.volume_energy_flux.sum(dim="sigma_layer")
    total_volume_per_column = ds.element_volume.sum(dim="sigma_layer")

    # Calculate volume-weighted average energy flux for each column
    column_volume_avg_energy_flux = (
        total_energy_flux_per_column / total_volume_per_column
    )

    # Add to dataset
    ds["column_volume_avg_energy_flux"] = column_volume_avg_energy_flux

    # Add metadata
    ds.column_volume_avg_energy_flux.attrs = {
        "long_name": "Column Volume-Weighted Average Energy Flux",
        "units": "W/m^3",
        "description": "Volume-weighted average energy flux for each vertical water column",
        "methodology": "Calculated as sum of energy flux over all sigma layers divided by sum of volumes for each face",
        "computation": 'column_volume_avg_energy_flux = sum(volume_energy_flux, dim="sigma_layer") / sum(element_volume, dim="sigma_layer")',
        "input_variables": "volume_energy_flux: energy flux in each element volume (W), element_volume: volume of water per element (m^3)",
        "visualization_purpose": "Provides a spatially-distributed metric of energy flux suitable for mapping, accounting for variable water depths and reducing the impact of outliers compared to simple speed averages",
    }

    return ds


def calculate_robust_column_energy_flux(ds, config=None, percentile=95):
    """
    Calculate a robust metric of energy flux for mapping that reduces the influence of outliers.

    This function computes a percentile-based energy flux for each vertical column,
    providing a metric that is less sensitive to extreme values than mean or maximum
    while still representing the high-energy parts of the water column.

    Parameters
    ----------
    ds : xarray.Dataset
        FVCOM dataset containing 'power_density' and 'element_volume'
    config : dict, optional
        Configuration dictionary
    percentile : float, optional
        Percentile to use for the calculation (default: 95)

    Returns
    -------
    xarray.Dataset
        Original dataset with added 'robust_column_energy_flux' variable
    """
    if "power_density" not in ds:
        raise KeyError(
            "Dataset must contain 'power_density'. Please calculate sea water power density first."
        )

    if "element_volume" not in ds:
        raise KeyError(
            "Dataset must contain 'element_volume'. Please calculate element volumes first."
        )

    # Get dimensions for computation
    n_time = len(ds.time)
    n_face = len(ds.face)

    # Initialize output array
    robust_energy_flux = np.zeros((n_time, n_face))

    # For each time step and face, calculate the volume-weighted percentile
    for t in range(n_time):
        for f in range(n_face):
            # Extract power density and volume for this column
            power_density_column = ds.power_density[t, :, f].values
            volume_column = ds.element_volume[t, :, f].values

            # Sort by power density
            sort_idx = np.argsort(power_density_column)
            sorted_power = power_density_column[sort_idx]
            sorted_volume = volume_column[sort_idx]

            # Calculate cumulative volume fraction
            cum_volume = np.cumsum(sorted_volume)
            total_volume = cum_volume[-1]
            volume_fraction = cum_volume / total_volume

            # Find the power density at the specified percentile
            # Use linear interpolation to find the exact percentile
            percentile_fraction = percentile / 100.0

            if percentile_fraction <= volume_fraction[0]:
                # Handle edge case where percentile is below first value
                robust_energy_flux[t, f] = sorted_power[0]
            elif percentile_fraction >= volume_fraction[-1]:
                # Handle edge case where percentile is above last value
                robust_energy_flux[t, f] = sorted_power[-1]
            else:
                # Find the two points to interpolate between
                idx = np.searchsorted(volume_fraction, percentile_fraction)
                v0, v1 = volume_fraction[idx - 1], volume_fraction[idx]
                p0, p1 = sorted_power[idx - 1], sorted_power[idx]

                # Linear interpolation
                interp_factor = (percentile_fraction - v0) / (v1 - v0)
                robust_energy_flux[t, f] = p0 + interp_factor * (p1 - p0)

    # Add to dataset
    ds["robust_column_energy_flux"] = xr.DataArray(
        robust_energy_flux,
        dims=["time", "face"],
        coords={"time": ds.time, "face": ds.face},
    )

    # Add metadata
    ds.robust_column_energy_flux.attrs = {
        "long_name": f"Volume-Weighted {percentile}th Percentile Energy Flux",
        "units": "W/m^2",
        "description": f"Volume-weighted {percentile}th percentile of power density in each vertical column",
        "methodology": f"Calculated as the volume-weighted {percentile}th percentile of power density values in each vertical column",
        "computation": "Calculated using volume-weighted percentile of power density values for each face",
        "input_variables": "power_density: sea water power density (W/m^2), element_volume: volume of water per element (m^3)",
        "visualization_purpose": "Provides a robust metric for mapping energy flux that reduces the influence of outliers while still representing high-energy regions",
    }

    return ds


#
# def calculate_zeta_center(ds):
#     """
#     Interpolate surface elevation (zeta) from nodes to face centers.
#
#     This function computes the surface elevation at cell centers by averaging
#     the nodal values that make up each triangular cell using the node-to-cell
#     connectivity array (nv).
#
#     Parameters
#     ----------
#     ds : xarray.Dataset
#         Dataset containing:
#         - 'zeta': surface elevation at nodes
#         - 'nv': node-to-cell connectivity array (1-based indices)
#
#     Returns
#     -------
#     xarray.Dataset
#         Original dataset with added 'zeta_center' variable and CF-compliant metadata
#
#     Raises
#     ------
#     KeyError
#         If required variables 'zeta' or 'nv' are missing
#     """
#     if not all(var in ds for var in ["zeta", "nv"]):
#         raise KeyError("Dataset must contain both 'zeta' and 'nv' variables")
#
#     # Get dimensions
#     n_times = ds.dims["time"]
#     n_faces = ds.dims["face"]
#
#     # Initialize output array
#     cell_values = np.zeros((n_times, n_faces))
#
#     # For each timestep, compute mean of nodal values for each cell
#     for t in range(n_times):
#         # Get the node indices for this timestep (subtract 1 to convert from 1-based to 0-based indexing)
#         node_indices = ds.nv[t].values - 1  # Shape will be (3, n_faces)
#
#         # Get nodal values for this timestep
#         node_values = ds.zeta[t].values
#
#         # For each face, get its three node values and average them
#         # We need to transpose node_indices to get shape (n_faces, 3)
#         face_node_values = node_values[node_indices.T]  # Shape: (n_faces, 3)
#         cell_values[t] = np.mean(face_node_values, axis=1)
#
#     # Add interpolated values to dataset
#     ds["zeta_center"] = xr.DataArray(
#         cell_values,
#         dims=["time", "cell"],
#         coords={"time": ds.zeta.time, "cell": np.arange(n_faces)},
#     )
#
#     # Copy and modify attributes from original zeta variable
#     ds.zeta_center.attrs = {
#         **ds.zeta.attrs,
#         "long_name": "Sea Surface Height at Cell Centers",
#         "coordinates": "time cell",
#         "mesh": "cell_centered",
#         "interpolation": (
#             "Computed by averaging the surface elevation values from the three "
#             "nodes that define each triangular cell using the node-to-cell "
#             "connectivity array (nv)."
#         ),
#         "computation": "zeta_center = mean(zeta[nv - 1], axis=1)",
#         "input_variables": "zeta: sea_surface_height_above_geoid at nodes",
#     }
#
#     return ds


def calculate_zeta_center(ds):
    # FVCOM is FORTRAN based and indexes start at 1
    # Convert indexes to python convention
    # nv = nv - 1
    nv = ds.nv
    nv = nv - 1

    # Reshape zeta to prepare for the operation
    # This creates a (time, 3, face) array where each face has its 3 node values
    zeta_at_nodes = ds.zeta.isel(node=nv)  # Should have shape (672, 3, 392002)
    print(zeta_at_nodes.shape)

    print("Computing zeta_center from zeta...")
    # Average along the node dimension (axis=1)
    zeta_center = zeta_at_nodes.mean(dim="face_node_index")

    print("Finished computing zeta_center!")
    # Add coordinates and attributes
    zeta_center = zeta_center.assign_coords(
        lon_center=ds.lon_center, lat_center=ds.lat_center
    )

    # Add attributes
    zeta_center.attrs = {
        **ds.zeta.attrs,
        "long_name": "Sea Surface Height at Cell Centers",
        "coordinates": "time cell",
        "mesh": "cell_centered",
        "interpolation": (
            "Computed by averaging the surface elevation values from the three "
            "nodes that define each triangular cell using the node-to-cell "
            "connectivity array (nv)."
        ),
        "computation": "zeta_center = mean(zeta[nv - 1], axis=1)",
        "input_variables": "zeta: sea_surface_height_above_geoid at nodes",
        # "units": "m",
        # "positive": "up",
        # "standard_name": "sea_surface_height_above_geoid",
        # "grid": "Bathymetry_Mesh",
        # "location": "face",
        # "coverage_content_type": "modelResult",
    }

    ds["zeta_center"] = zeta_center

    print(ds.zeta_center)
    print("Max: ", ds.zeta_center.max())
    print("Min: ", ds.zeta_center.min())
    print("Mean: ", ds.zeta_center.mean())

    return ds


# def calculate_zeta_center(ds, max_node_difference=0.1):
#     """
#     Calculate water surface elevation at cell centers by averaging the values at the three
#     surrounding nodes of each face.
#
#     Parameters:
#     -----------
#     ds : xarray.Dataset
#         Dataset containing zeta (at nodes) and nv (face-node connectivity) variables
#     max_node_difference : float, optional
#         Maximum allowed difference in meters between any two nodes in a cell.
#         Default is 0.1 meters, which is reasonable for most tidal flows.
#
#     Returns:
#     --------
#     xarray.Dataset
#         Input dataset with new zeta_center variable added
#
#     Raises:
#     -------
#     ValueError
#         If node indexing is unexpected or if node differences exceed threshold
#     """
#
#     # Verify the indexing and convert if needed
#     nv_min = ds["nv"].min().item()
#     if nv_min == 1:  # Confirms 1-based indexing
#         # Convert from 1-based to 0-based indexing
#         node_indices = ds["nv"].values.T - nv_min  # Shape: (nele, 3, time)
#     else:
#         raise ValueError(
#             f"Unexpected minimum node index in nv: {nv_min}. Expected 1 for FVCOM 1-based indexing."
#         )
#
#     # Get zeta values as numpy array
#     zeta_values = ds["zeta"].values  # Shape: (time, node)
#
#     # Create array to store zeta values for each node of each face
#     # Reshape zeta to (1, time, node) for broadcasting
#     zeta_reshaped = zeta_values.reshape(zeta_values.shape[0], 1, -1)
#
#     # Use advanced indexing to get values for each node
#     # node_indices shape: (nele, 3, time)
#     # We want final shape: (time, nele, 3)
#     zeta_at_nodes = np.take_along_axis(
#         zeta_reshaped,
#         node_indices.transpose(2, 0, 1),  # Transpose to (time, nele, 3)
#         axis=2,
#     )
#
#     # Validate node-to-node differences within each cell
#     # Calculate pairwise differences between nodes
#     diff_01 = np.abs(zeta_at_nodes[..., 0] - zeta_at_nodes[..., 1])
#     diff_12 = np.abs(zeta_at_nodes[..., 1] - zeta_at_nodes[..., 2])
#     diff_20 = np.abs(zeta_at_nodes[..., 2] - zeta_at_nodes[..., 0])
#
#     # Find maximum difference for each cell
#     max_diff = np.maximum.reduce([diff_01, diff_12, diff_20])
#
#     # Check for problems
#     if np.any(max_diff > max_node_difference):
#         n_problems = np.sum(max_diff > max_node_difference)
#         max_found = np.max(max_diff)
#         raise ValueError(
#             f"Found {n_problems} cells with node-to-node elevation differences "
#             f"exceeding {max_node_difference}m threshold. Maximum difference "
#             f"found: {max_found:.3f}m. This might indicate numerical instabilities "
#             "or areas of strong gradients (e.g., tidal rapids)."
#         )
#
#     # Average the three nodes to get the cell center value
#     zeta_center_values = np.mean(zeta_at_nodes, axis=2)  # Shape: (time, nele)
#
#     # Create new DataArray with same dimensions as h_center
#     zeta_center = xr.DataArray(
#         zeta_center_values,
#         dims=ds.h_center.dims,
#         coords=ds.h_center.coords,
#         attrs={
#             **ds.zeta.attrs,
#             "long_name": "Sea Surface Height at Cell Centers",
#             "coordinates": "time cell",
#             "mesh": "cell_centered",
#             "interpolation": (
#                 "Computed by averaging the surface elevation values from the three "
#                 "nodes that define each triangular cell using the node-to-cell "
#                 "connectivity array (nv)."
#             ),
#             "computation": "zeta_center = mean(zeta[nv - 1], axis=1)",
#             "input_variables": "zeta: sea_surface_height_above_geoid at nodes",
#             "validation_threshold": f"Maximum allowed node-to-node difference: {max_node_difference}m",
#         },
#     )
#
#     # Add the new variable to the dataset
#     ds["zeta_center"] = zeta_center
#     return ds


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
    required_vars = ["h_center", "zeta_center"]
    if not all(var in ds for var in required_vars):
        raise KeyError(
            "Dataset must contain both 'h_center' (bathymetry) and 'zeta_center' (surface elevation)"
        )

    # Verify h_center attributes
    if "units" not in ds.h_center.attrs:
        raise ValueError("Input dataset `h_center` units must be defined!")
    if "positive" not in ds.h_center.attrs:
        raise ValueError("Input dataset `h_center` must define positive direction!")
    if ds.h_center.attrs["positive"] != "down":
        raise ValueError("Input dataset `h_center` positive direction must be 'down'!")

    # Verify zeta_center attributes
    if "units" not in ds.zeta_center.attrs:
        raise ValueError("Input dataset `zeta_center` units must be defined!")
    if "positive" not in ds.zeta_center.attrs:
        raise ValueError("Input dataset `zeta_center` must define positive direction!")
    if ds.zeta_center.attrs["positive"] != "up":
        raise ValueError("Input dataset `zeta_center` positive direction must be 'up'!")

    # Verify units match
    h_units = ds.h_center.attrs["units"]
    z_units = ds.zeta_center.attrs["units"]
    if h_units != z_units:
        raise ValueError(f"Units mismatch: h_center: {h_units}, zeta_center: {z_units}")


def calculate_depth(ds):
    """
    Calculate depth at sigma levels for FVCOM output.

    This function computes the depth at each sigma layer using the equation:
    depth = -(h + ζ) * σ

    where:
        h: bathymetry (positive down from geoid)
        ζ: sea surface elevation (positive up from geoid)
        σ: sigma layer coordinates (negative, surface to bottom)

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing:
        - 'h_center': bathymetry at cell centers
        - 'zeta_center': water surface elevation at cell centers
        - 'sigma': sigma layer coordinates

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
        raise KeyError("Dataset must contain 'sigma_level' coordinates")

    # Extract one sigma layer
    sigma_layer = ds.sigma_layer.T.values[0]

    # Calculate depth at each sigma level
    # ds["depth"] = -(ds.h_center + ds.zeta_center) * sigma_layer

    # Reshape sigma to (1, sigma_layer, 1) for proper broadcasting
    sigma_3d = sigma_layer.reshape(1, -1, 1)

    # Calculate total water depth (h + zeta) and reshape
    total_depth = ds.h_center + ds.zeta_center  # Shape (time, face)
    total_depth_3d = total_depth.expand_dims(
        dim={"sigma_layer": len(sigma_layer)}, axis=1
    )

    # Calculate depth
    ds["depth"] = -(total_depth_3d * sigma_3d)

    # Add CF-compliant metadata
    ds.depth.attrs = {
        "long_name": "Depth Below Sea Surface",
        "standard_name": "depth",
        "units": ds.h_center.attrs["units"],
        "positive": "down",
        "coordinates": "time cell sigma",
        "description": (
            "Depth represents the vertical distance below the sea surface at each sigma "
            "layer, varying with both the fixed bathymetry and time-varying surface elevation."
        ),
        "methodology": (
            "Depth is calculated using the sigma coordinate transformation: "
            "depth = -(h + ζ) * σ, where h is bathymetry, ζ is surface elevation, "
            "and σ is the sigma layer coordinate."
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
    # validate_depth_inputs(ds)

    # Calculate total water column depth
    ds["seafloor_depth"] = -(ds.h_center) + ds.zeta_center

    # Add CF-compliant metadata
    ds.seafloor_depth.attrs = {
        "long_name": "Sea Floor Depth Below Sea Surface",
        "standard_name": "sea_floor_depth_below_sea_surface",
        "units": ds.h_center.attrs["units"],
        "positive": "down",
        "description": (
            "The vertical distance between the sea surface and the seabed as measured "
            "at a given point in space including the variance caused by tides."
        ),
        "methodology": (
            "Total water column depth is calculated by combining the fixed bathymetry "
            "and time-varying surface elevation. Since h_center is stored as negative "
            "values (depth below geoid) and zeta_center as positive (height above geoid), "
            "we negate h_center and add zeta_center to get the total depth."
        ),
        "computation": "seafloor_depth = -(h_center) + zeta_center",
        "input_variables": (
            "h_center: sea_floor_depth_below_geoid (m), "
            "zeta_center: sea_surface_height_above_geoid (m)"
        ),
    }

    return ds


def calculate_vertical_average(ds, variable_name):
    """
    Calculate vertical average for a given variable and include standard deviation in attributes.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing the variable to be averaged
    variable_name : str
        Name of variable to average vertically

    Returns
    -------
    xarray.Dataset
        Dataset with added vertically averaged variable
    """
    if variable_name not in ds:
        raise KeyError(f"Dataset must contain '{variable_name}'")

    # Calculate vertical average and standard deviation
    vert_avg_name = f"{variable_name}_vert_avg"

    ds[vert_avg_name] = ds[variable_name].mean(dim="sigma_layer")
    vert_std = ds[variable_name].std(dim="sigma_layer")

    # Copy and modify attributes for averaged variable
    # Start with original attributes but remove standard_name if it exists
    attrs = ds[variable_name].attrs.copy()
    attrs.pop("standard_name", None)

    ds[vert_avg_name].attrs = {
        **attrs,
        "long_name": f"Vertically averaged {ds[variable_name].attrs.get('long_name', variable_name)}",
        "vertical_averaging": "Mean across sigma layers",
        "vertical_std": float(vert_std.mean()),  # Average std across all points/times
        "vertical_std_description": "Mean standard deviation across sigma layers",
    }

    return ds


def derive_vap(config, location_key):
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

    for count, nc_file in enumerate(std_partition_nc_files, start=1):
        print(f"Calculating vap for {nc_file}")
        this_ds = xr.open_dataset(nc_file)

        print("\tCalculating speed...")
        this_ds = calculate_sea_water_speed(this_ds, config)
        print("\tCalculating direction...")
        this_ds = calculate_sea_water_direction(this_ds, config)
        print("\tCalculating power density...")
        this_ds = calculate_sea_water_power_density(this_ds, config)
        print("\tCalculating zeta_center...")
        this_ds = calculate_zeta_center(this_ds)

        print("\tCalculating depth...")
        this_ds = calculate_depth(this_ds)
        print("\tCalculating sea_floor_depth...")
        this_ds = calculate_sea_floor_depth(this_ds)

        print("\tCalculating element volumes...")
        this_ds = calculate_element_volumes(this_ds)

        print("\tCalculating element volumes...")
        this_ds = calculate_element_volumes(this_ds)

        print("\tCalculating volume energy flux...")
        this_ds = calculate_volume_energy_flux(this_ds)

        print("\tCalculating vertical avg energy flux...")
        this_ds = calculate_vertical_avg_energy_flux(this_ds)

        print("\tCalculating volume avg energy flux...")
        this_ds = calculate_volume_avg_energy_flux(this_ds)

        print("\tCalculating column avg energy flux...")
        this_ds = calculate_column_volume_avg_energy_flux(this_ds)

        print("\tCalculating robust column avg energy flux...")
        this_ds = calculate_robust_column_energy_flux(this_ds)

        print("\tCalculating u vertical average")
        this_ds = calculate_vertical_average(this_ds, "u")
        print("\tCalculating v vertical average")
        this_ds = calculate_vertical_average(this_ds, "v")
        print("\tCalculating speed vertical average")
        this_ds = calculate_vertical_average(this_ds, "speed")
        print("\tCalculating from_direction vertical average")
        this_ds = calculate_vertical_average(this_ds, "from_direction")
        print("\tCalculating power_density vertical average")
        this_ds = calculate_vertical_average(this_ds, "power_density")

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
            file_manager.get_vap_output_dir(config, location),
            f"{count:03d}.{data_level_file_name}",
        )

        print(f"\tSaving to {output_path}...")
        this_ds.to_netcdf(output_path, encoding=config["dataset"]["encoding"])

        this_ds.close()
        gc.collect()
