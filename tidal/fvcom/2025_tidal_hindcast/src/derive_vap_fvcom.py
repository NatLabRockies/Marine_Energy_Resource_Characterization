import gc

from pathlib import Path

import numpy as np
import xarray as xr

from . import file_manager, file_name_convention_manager


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


def calculate_sea_water_speed(ds):
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

    # Calculate speed maintaining original dimensions
    ds["speed"] = np.sqrt(ds.u**2 + ds.v**2)

    # Add CF-compliant metadata
    ds.sea_water_speed.attrs = {
        "long_name": "Sea Water Speed",
        "standard_name": "sea_water_speed",
        "units": ds.u.attrs.get("units", "m s-1"),
        "grid": ds.u.attrs.get("grid"),
        "type": "data",
        "mesh": ds.u.attrs.get("mesh"),
        "location": ds.u.attrs.get("location"),
        "description": "Speed is the magnitude of velocity.",
        "methodology": (
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


def calculate_sea_water_direction(ds, direction_undefined_speed_threshold_ms=0.0):
    """
    Calculate sea water velocity direction in meteorological convention.

    This function computes the direction water is coming FROM (like wind direction),
    measured clockwise from true north. The calculation involves:
    1. Converting velocity components to mathematical angle (counterclockwise from east)
    2. Converting to meteorological convention (clockwise from north)
    3. Converting to 'from' direction (adding 180° to get source direction)

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing:
        - 'u': eastward sea water velocity component
        - 'v': northward sea water velocity component
        - 'sea_water_speed': magnitude of velocity (must be pre-computed)
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

    if "sea_water_speed" not in ds.variables:
        raise KeyError(
            "Dataset must contain 'sea_water_speed'. "
            "Please run calculate_sea_water_speed() first."
        )

    # Calculate mathematical direction (counterclockwise from east)
    mathematical_direction_degrees = np.rad2deg(np.arctan2(ds.v, ds.u))

    # Validate direction ranges
    mathematical_direction_degrees_expected_max = 360.0
    mathematical_direction_degrees_expected_min = 0

    mathematical_direction_degrees_max = np.max(mathematical_direction_degrees)
    mathematical_direction_degrees_min = np.min(mathematical_direction_degrees)

    if mathematical_direction_degrees_max > mathematical_direction_degrees_expected_max:
        raise ValueError(
            f"Maximum direction value {mathematical_direction_degrees_max}° "
            f"exceeds expected maximum of {mathematical_direction_degrees_expected_max}°"
        )

    if mathematical_direction_degrees_min > mathematical_direction_degrees_expected_min:
        raise ValueError(
            f"Minimum direction value {mathematical_direction_degrees_min}° "
            f"exceeds expected minimum of {mathematical_direction_degrees_expected_min}°"
        )

    # Convert to meteorological convention (clockwise from north)
    meterological_to_direction_degrees = np.mod(
        mathematical_direction_degrees - 90.0, 360.0
    )

    # Convert to 'from' direction (direction water comes from)
    meteorological_from_direction_degrees = np.mod(
        meterological_to_direction_degrees - 180.0, 360.0
    )

    # Set directions to NaN where speed is below threshold
    meteorological_from_direction_degrees = xr.where(
        ds.sea_water_speed > direction_undefined_speed_threshold_ms,
        meteorological_from_direction_degrees,
        np.nan,
    )

    ds["from_direction"] = meteorological_from_direction_degrees

    # Add CF-compliant metadata
    ds.sea_water_velocity_from_direction.attrs = {
        "long_name": "Sea Water Velocity From Direction",
        "standard_name": "sea_water_velocity_from_direction",
        "units": "degree",
        "valid_min": "0.0",  # Convert to string for consistency
        "valid_max": "360.0",  # Convert to string for consistency
        "grid": ds.u.attrs.get("grid"),
        "type": "data",
        "mesh": ds.u.attrs.get("mesh"),
        "location": ds.u.attrs.get("location"),
        # From CF Standard Name
        "description": (
            "A velocity is a vector quantity. ",
            'The phrase "from_direction" is used in the construction X_from_direction '
            "and indicates the direction from which the velocity vector of X is coming. "
            "The direction is a bearing in the usual geographical sense, "
            "measured positive clockwise from due north.",
        ),
        "methodology": (
            "Direction is calculated in three steps: 1. Calculate mathematical direction "
            "using arctan2(v,u), 2. Convert to meteorological convention by subtracting "
            "from 90°, 3. Convert to 'from' direction by subtracting 180°"
            "Directions are set to NaN for "
            f"speeds below {direction_undefined_speed_threshold_ms} m/s."
        ),
        "computation": (
            "mathematical_direction_degrees = np.rad2deg(np.arctan2(ds.v, ds.u))\n"
            "meterological_to_direction_degrees = np.mod(mathematical_direction_degrees - 90.0, 360.0)\n"
            "meteorological_from_direction_degrees = np.mod(meterological_to_direction_degrees - 180.0, 360.0)"
        ),
        "input_variables": (
            "u: eastward_sea_water_velocity (m/s), "
            "v: northward_sea_water_velocity (m/s)"
        ),
    }

    return ds


def calculate_sea_water_power_density(ds, rho: float = 1025.0):
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

    if "sea_water_speed" not in ds:
        raise KeyError(
            "Dataset must contain 'sea_water_speed'. "
            "Please run calculate_sea_water_speed() first."
        )

    # Calculate power density using Equation 1 from
    # Haas, Kevin A., et al. "Assessment of Energy Production Potential from
    # Tidal Streams in the United States." , Jun. 2011. https://doi.org/10.2172/1219367
    ds["power_density"] = 0.5 * rho * ds.sea_water_speed**3

    # Add CF-compliant metadata
    ds.sea_water_power_density.attrs = {
        "long_name": "Sea Water Power Density",
        "units": "W m-2",
        "grid": ds.sea_water_speed.attrs.get("grid"),
        "type": "data",
        "mesh": ds.sea_water_speed.attrs.get("mesh"),
        "location": ds.sea_water_speed.attrs.get("location"),
        "methodology": (
            "Computed using the fluid power density equation P = ½ρv³ with seawater "
            f"density ρ = {rho} kg/m³. The calculation uses sea water speed derived "
            "from FVCOM u,v velocity components written to the sea_water_speed variable."
        ),
        "computation": "sea_water_power_density = 0.5 * rho * sea_water_speed**3",
        "input_variables": ("sea_water_speed (m/s), rho (kg/m³)"),
        "citation": (
            "Haas, Kevin A., et al. 'Assessment of Energy Production Potential "
            "from Tidal Streams in the United States.' Georgia Tech Research "
            "Corporation, Jun. 2011. https://doi.org/10.2172/1219367"
        ),
    }

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

    if "sigma" not in ds:
        raise KeyError("Dataset must contain 'sigma' coordinates")

    # Calculate depth at each sigma level
    ds["depth"] = -(ds.h_center + ds.zeta_center) * ds.sigma

    # Add CF-compliant metadata
    ds.depth.attrs = {
        "long_name": "Depth Below Sea Surface",
        "standard_name": "depth",
        "units": ds.h_center.attrs["units"],
        "positive": "down",
        "grid": ds.h_center.attrs.get("grid"),
        "grid_location": "center",
        "mesh": ds.h_center.attrs.get("mesh"),
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
    validate_depth_inputs(ds)

    # Calculate total water column depth
    ds["seafloor_depth"] = -(ds.h_center) + ds.zeta_center

    # Add CF-compliant metadata
    ds.seafloor_depth.attrs = {
        "long_name": "Sea Floor Depth Below Sea Surface",
        "standard_name": "sea_floor_depth_below_sea_surface",
        "units": ds.h_center.attrs["units"],
        "positive": "down",
        "grid": ds.h_center.attrs.get("grid"),
        "grid_location": "center",
        "mesh": ds.h_center.attrs.get("mesh"),
        "coordinates": "time cell",
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


def derive_vap(config, location_key):
    location = config[location_key]

    std_partition_path = file_manager.get_standardized_partition_output_dir(
        config, location
    )
    std_partition_nc_files = sorted(list(std_partition_path.rglob("*.nc")))

    for count, nc_file in enumerate(std_partition_nc_files, start=1):
        this_ds = xr.open_dataset(nc_file)

        this_ds = calculate_sea_water_speed(this_ds)
        this_ds = calculate_sea_water_direction(this_ds)
        this_ds = calculate_sea_water_power_density(this_ds)
        this_ds = calculate_depth(this_ds)
        this_ds = calculate_sea_floor_depth(this_ds)

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
                config["dataset_name"],
                "b1",
                temporal=temporal_string,
            )
        )

        output_path = Path(
            file_manager.get_vap_output_dir(config, location),
            f"{count:03d}.{data_level_file_name}",
        )

        this_ds.to_netcdf(output_path)

        this_ds.close()
        gc.collect()
