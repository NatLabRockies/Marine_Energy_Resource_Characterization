import cProfile
import gc
import io
import os
import pstats
import time

from pathlib import Path

import dask
import dask.array as da
import numpy as np
import psutil
import xarray as xr

from dask.diagnostics import ProgressBar

from . import attrs_manager, file_manager, file_name_convention_manager


def get_memory_mb():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def profile_function(func):
    """
    Decorator to profile a function's execution time and memory usage.

    Parameters
    ----------
    func : function
        Function to profile

    Returns
    -------
    function
        Wrapper function that profiles execution
    """

    def wrapper(*args, **kwargs):
        # Get memory before execution
        mem_before = get_memory_mb()

        # Set up profiler
        profiler = cProfile.Profile()
        profiler.enable()
        start_time = time.time()

        # Execute the function
        result = func(*args, **kwargs)

        # Get execution time
        end_time = time.time()
        elapsed_time = end_time - start_time

        # Get memory after execution
        mem_after = get_memory_mb()
        mem_diff = mem_after - mem_before

        # Disable profiler
        profiler.disable()

        # Print performance summary
        print(f"\n### PROFILING: {func.__name__} ###")
        print(f"Time: {elapsed_time:.2f} seconds")
        print(f"Memory: {mem_before:.1f} MB → {mem_after:.1f} MB (Δ {mem_diff:.1f} MB)")

        # Get detailed stats
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats("cumulative")
        ps.print_stats(20)  # Print top 20 functions by cumulative time

        # Print formatted stats
        print(s.getvalue())

        return result

    return wrapper


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

    if "speed" not in ds.variables:
        raise KeyError(
            "Dataset must contain 'speed'. "
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
        ds.speed > direction_undefined_speed_threshold_ms,
        compass_to_direction_degrees,
        np.nan,
    )

    output_variable_name = "to_direction"

    ds[output_variable_name] = compass_to_direction_degrees

    specified_attrs = config["derived_vap_specification"][output_variable_name][
        "attributes"
    ]

    # Add CF-compliant metadata
    ds.to_direction.attrs = {
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
    if "to_direction" not in ds.variables:
        ds = calculate_sea_water_to_direction(
            ds, config, direction_undefined_speed_threshold_ms
        )

    # Convert compass 'to' direction to 'from' direction by adding 180°
    # Example: u=1, v=0 (east) has to_direction 90°
    # 90 + 180 = 270° (coming from west)
    compass_from_direction_degrees = np.mod(ds.to_direction + 180, 360)

    # Set directions to NaN where speed is below threshold
    if "speed" in ds.variables:
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
    nv = ds.nv.isel(time=0).compute().values - 1  # Convert to 0-based indexing

    lon_node = ds.lon_node.compute().values
    lat_node = ds.lat_node.compute().values

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
    # Calculate element areas - these are constant for all time steps
    element_areas = calculate_element_areas(ds)

    # Get dimensions
    # n_time = len(ds.time)
    n_sigma_layer = len(ds.sigma_layer)
    n_face = len(ds.face)

    # Get node indices for each element (face) - first time index only
    # Need to compute to avoid Dask indexing issues
    nv = ds.nv.isel(time=0).compute().values - 1  # Convert to 0-based indexing

    # Get sigma layer and level values - these are usually small arrays
    # sigma_layer_values = ds.sigma_layer.compute().values
    sigma_level_values = ds.sigma_level.compute().values

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

    total_depth = abs(ds.h_center) + ds.zeta_center

    # First create a DataArray for element_areas
    element_areas_da = xr.DataArray(
        element_areas, dims=["face"], coords={"face": ds.face}
    )

    # Create a DataArray for element_thickness_fraction
    element_thickness_da = xr.DataArray(
        element_thickness_fraction,
        dims=["sigma_layer", "face"],
        coords={"sigma_layer": ds.sigma_layer, "face": ds.face},
    )

    # Calculate volumes using xarray operations to maintain Dask arrays
    element_volumes = element_areas_da * total_depth * element_thickness_da

    # Add element volumes to dataset
    ds["element_volume"] = element_volumes.assign_attrs(
        {
            "long_name": "Element Volume",
            "standard_name": "volume_of_water_per_element",
            "units": "m^3",
            "description": "Volume of each triangular element at each time step and sigma layer",
            "methodology": "Calculated as element area multiplied by layer thickness using fully vectorized operations",
            "computation": "element_volume = element_area * total_water_depth * sigma_layer_thickness",
            "input_variables": "h_center: bathymetry (m), zeta_center: surface elevation (m), sigma_layer: sigma coordinate",
        }
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
    volume_energy_flux = ds.power_density * ds.element_volume

    # Add volume energy flux to dataset
    ds["volume_energy_flux"] = volume_energy_flux

    # Add metadata
    ds.volume_energy_flux.attrs = {
        "long_name": "Model Element Volume Energy Flux",
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


def calculate_column_volume_avg_energy_flux(ds):
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


@profile_function
def calculate_zeta_center(ds):
    """
    Calculate sea surface height at cell centers.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing zeta and nv variables

    Returns
    -------
    xarray.Dataset
        Dataset with added zeta_center variable
    """
    # FVCOM is FORTRAN based and indexes start at 1
    # Convert indexes to python convention and fully compute
    print("\t\tComputing nv array...")
    nv = ds.nv.compute() - 1

    # Fully compute zeta array
    print("\t\tComputing zeta array...")
    zeta = ds.zeta.compute()

    # Perform the indexing operation using computed arrays
    print("\t\tPerforming indexing operation...")
    zeta_at_nodes = zeta.isel(node=nv)

    # Average along the node dimension
    print("\t\tCalculating mean...")
    zeta_center = zeta_at_nodes.mean(dim="face_node_index")

    # Add coordinates and attributes
    print("\t\tAdding coordinates...")
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
    }

    # Add to dataset
    ds["zeta_center"] = zeta_center

    # Clear any cached computations to free memory
    print("\t\tClearing cached data...")
    zeta = None
    zeta_at_nodes = None

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
    sigma_layer = ds.sigma_layer.compute().T.values[0]

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


def calculate_all_vertical_stats(ds, variable_name):
    """
    Calculate all vertical statistics (average, median, 95th percentile) for a variable
    in a single pass to improve performance.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing the variable to be processed
    variable_name : str
        Name of variable to calculate statistics for

    Returns
    -------
    xarray.Dataset
        Dataset with added statistical variables
    """
    if variable_name not in ds:
        print(f"\tSkipping {variable_name} - not found in dataset")
        return ds

    print(f"\tCalculating {variable_name} vertical average")
    print(f"\tCalculating {variable_name} vertical median")
    print(f"\tCalculating {variable_name} vertical 95th_percentile")

    # Get original variable and its attributes
    var = ds[variable_name]
    orig_attrs = var.attrs.copy()

    # Create a common function for all three calculations
    # This is more efficient than calculating them separately
    def calc_stats(x):
        # Compute mean, median and 95th percentile at once
        # This is faster than three separate calls when using dask
        mean_val = np.mean(x, axis=0)

        # Sort once for both median and percentile
        sorted_x = np.sort(x, axis=0)
        n = x.shape[0]

        # Extract median
        if n % 2 == 0:
            median_val = (sorted_x[n // 2 - 1] + sorted_x[n // 2]) / 2
        else:
            median_val = sorted_x[n // 2]

        # Extract 95th percentile
        idx = int(np.ceil(0.95 * n) - 1)
        p95_val = sorted_x[idx]

        return mean_val, median_val, p95_val

    # Apply the function to compute all statistics at once
    if hasattr(var.data, "map_blocks") and isinstance(var.data, da.Array):
        # For dask arrays, use map_blocks to maintain chunking
        mean_data, median_data, p95_data = da.apply_along_axis(
            lambda x: calc_stats(x),
            axis=var.get_axis_num("sigma_layer"),
            arr=var.data,
            dtype=var.dtype,
        )
    else:
        # For numpy arrays, use apply_along_axis directly
        results = np.apply_along_axis(
            calc_stats, axis=var.get_axis_num("sigma_layer"), arr=var.values
        )
        mean_data, median_data, p95_data = results

    # Create the mean DataArray with appropriate metadata
    vert_avg_name = f"{variable_name}_vert_avg"
    ds[vert_avg_name] = xr.DataArray(
        mean_data,
        dims=var.dims[1:],  # Remove sigma_layer dimension
        coords={k: var.coords[k] for k in var.dims if k != "sigma_layer"},
    )

    # Set mean attributes
    mean_attrs = orig_attrs.copy()
    mean_attrs.pop("standard_name", None)
    ds[vert_avg_name].attrs = {
        **mean_attrs,
        "long_name": f"Vertically averaged {orig_attrs.get('long_name', variable_name)}",
        "vertical_averaging": "Mean across sigma layers",
    }

    # Create the median DataArray with appropriate metadata
    median_name = f"{variable_name}_median"
    ds[median_name] = xr.DataArray(
        median_data,
        dims=var.dims[1:],  # Remove sigma_layer dimension
        coords={k: var.coords[k] for k in var.dims if k != "sigma_layer"},
    )

    # Set median attributes
    ds[median_name].attrs = {
        "long_name": f"Vertical median of {orig_attrs.get('long_name', variable_name)}",
        "units": orig_attrs.get("units", ""),
        "additional_processing": "Median calculated along the sigma_layer dimension.",
        "computation": f"median_values = ds['{variable_name}'].median(dim='sigma_layer')",
        "input_variables": f"{variable_name}: original variable",
    }

    # Create the 95th percentile DataArray with appropriate metadata
    p95_name = f"{variable_name}_vert_95th_percentile"
    ds[p95_name] = xr.DataArray(
        p95_data,
        dims=var.dims[1:],  # Remove sigma_layer dimension
        coords={k: var.coords[k] for k in var.dims if k != "sigma_layer"},
    )

    # Set 95th percentile attributes
    ds[p95_name].attrs = {
        "long_name": f"95th percentile of {orig_attrs.get('long_name', variable_name)}",
        "units": orig_attrs.get("units", ""),
        "additional_processing": "95th percentile calculated along the sigma_layer dimension.",
        "computation": f"percentile_95_values = ds['{variable_name}'].quantile(0.95, dim='sigma_layer')",
        "input_variables": f"{variable_name}: original variable",
    }

    return ds


def calculate_vertical_statistics(ds):
    """
    Calculate all vertical statistics for relevant variables in one pass
    while preserving individual print statements.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to calculate statistics for

    Returns
    -------
    xarray.Dataset
        Dataset with added statistical variables
    """
    # Variables that need all statistics
    full_stat_vars = ["speed", "power_density", "volume_energy_flux"]

    # Variables that need only average
    avg_only_vars = ["u", "v", "to_direction", "from_direction"]

    # Process variables needing all statistics
    for var in full_stat_vars:
        if var in ds:
            ds = calculate_all_vertical_stats(ds, var)

    # Process variables needing only average
    for var in avg_only_vars:
        if var in ds:
            print(f"\tCalculating {var} vertical average")
            ds = calculate_vertical_average(ds, var)

    return ds


# Keep the original functions for backward compatibility
def calculate_vertical_average(ds, variable_name):
    """
    Calculate vertical average for a given variable

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

    # Copy and modify attributes for averaged variable
    # Start with original attributes but remove standard_name if it exists
    attrs = ds[variable_name].attrs.copy()
    attrs.pop("standard_name", None)

    ds[vert_avg_name].attrs = {
        **attrs,
        "long_name": f"Vertically averaged {ds[variable_name].attrs.get('long_name', variable_name)}",
        "vertical_averaging": "Mean across sigma layers",
    }

    return ds


def calculate_vertical_median(ds, variable_name):
    """
    Calculate the median value of a variable along a specified dimension.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset with the variable to calculate median for
    variable_name : str
        Name of the variable to calculate median for

    Returns
    -------
    xarray.Dataset
        Input dataset with added median variable
    """
    if variable_name not in ds.variables:
        raise KeyError(
            f"Dataset must contain '{variable_name}'. "
            f"Please ensure this variable exists in the dataset."
        )

    dim = "sigma_layer"

    # Calculate median along the specified dimension
    median_values = ds[variable_name].median(dim=dim)

    output_variable_name = f"{variable_name}_median"

    ds[output_variable_name] = median_values

    # Add basic metadata
    ds[output_variable_name].attrs = {
        "long_name": f"Vertical median of {ds[variable_name].attrs.get('long_name', variable_name)}",
        "units": ds[variable_name].attrs.get("units", ""),
        "additional_processing": (f"Median calculated along the {dim} dimension."),
        "computation": (f"median_values = ds['{variable_name}'].median(dim='{dim}')\n"),
        "input_variables": (f"{variable_name}: original variable"),
    }

    return ds


def calculate_vertical_95th_percentile(ds, variable_name):
    """
    Calculate the 95th percentile value of a variable along a specified dimension.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset with the variable to calculate 95th percentile for
    variable_name : str
        Name of the variable to calculate 95th percentile for

    Returns
    -------
    xarray.Dataset
        Input dataset with added 95th percentile variable
    """
    if variable_name not in ds.variables:
        raise KeyError(
            f"Dataset must contain '{variable_name}'. "
            f"Please ensure this variable exists in the dataset."
        )

    dim = "sigma_layer"

    # Calculate 95th percentile along the specified dimension
    percentile_95_values = ds[variable_name].quantile(0.95, dim=dim)

    output_variable_name = f"{variable_name}_vert_95th_percentile"

    ds[output_variable_name] = percentile_95_values

    # Add basic metadata
    ds[output_variable_name].attrs = {
        "long_name": f"95th percentile of {ds[variable_name].attrs.get('long_name', variable_name)}",
        "units": ds[variable_name].attrs.get("units", ""),
        "additional_processing": (
            f"95th percentile calculated along the {dim} dimension."
        ),
        "computation": (
            f"percentile_95_values = ds['{variable_name}'].quantile(0.95, dim='{dim}')\n"
        ),
        "input_variables": (f"{variable_name}: original variable"),
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

    # Set xarray to preserve attributes during operations
    xr.set_options(keep_attrs=True)

    for count, nc_file in enumerate(std_partition_nc_files, start=1):
        print(f"Calculating vap for {nc_file}")

        # Define Dask chunking strategy
        chunks = {
            "time": "auto",  # Process all timesteps together
            "face": "auto",  # Let dask automatically determine chunk size for face
        }

        this_ds = xr.open_dataset(nc_file, chunks=chunks)

        print("\tCalculating speed...")
        this_ds = calculate_sea_water_speed(this_ds, config)

        print("\tCalculating to direction...")
        this_ds = calculate_sea_water_to_direction(this_ds, config)

        print("\tCalculating from direction...")
        this_ds = calculate_sea_water_from_direction(this_ds, config)

        print("\tCalculating power density...")
        this_ds = calculate_sea_water_power_density(this_ds, config)

        print("\tCalculating zeta_center...")
        this_ds = calculate_zeta_center(this_ds)

        print("\tCalculating depth...")
        this_ds = calculate_depth(this_ds)

        print("\tCalculating sea_floor_depth...")
        this_ds = calculate_sea_floor_depth(this_ds)

        print("\tCalculating element volumes...")
        this_ds = calculate_element_volume(this_ds)

        print("\tCalculating volume energy flux...")
        this_ds = calculate_volume_energy_flux(this_ds)

        print("\tCalculating vertical avg energy flux...")
        this_ds = calculate_vertical_avg_energy_flux(this_ds)

        print("\tCalculating column avg energy flux...")
        this_ds = calculate_column_volume_avg_energy_flux(this_ds)

        # Use the optimized vertical statistics calculation
        # This will generate all the same print statements as before
        # but perform the calculations more efficiently
        this_ds = calculate_vertical_statistics(this_ds)

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

        # Add progress bar for the computation phase
        with ProgressBar():
            this_ds.to_netcdf(output_path, encoding=config["dataset"]["encoding"])

        # Clean up to free memory
        this_ds.close()
        gc.collect()
