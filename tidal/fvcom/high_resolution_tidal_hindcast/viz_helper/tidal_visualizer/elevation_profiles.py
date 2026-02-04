# elevation_profiles module - Part of tidal_visualizer package

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.interpolate import griddata, interp1d
from shapely.geometry import LineString
import matplotlib.patheffects as PathEffects

from .utils import haversine_distance


def create_elevation_profile(
    ds, start_point, end_point, variable_name=None, num_points=100
):
    """
    Create an elevation/bathymetry profile between two points.

    Parameters:
    -----------
    ds : xarray.Dataset
        Dataset containing bathymetry data
    start_point : tuple
        (lat, lon) of the starting point
    end_point : tuple
        (lat, lon) of the ending point
    variable_name : str, optional
        Name of a variable to plot on top of the bathymetry
    num_points : int, optional
        Number of points to sample along the profile

    Returns:
    --------
    matplotlib.figure.Figure
        Figure containing the elevation profile
    """
    # Extract bathymetry data
    if "seafloor_depth" in ds:
        depth_var = "seafloor_depth"
    elif "depth" in ds:
        depth_var = "depth"
    elif "bathymetry" in ds:
        depth_var = "bathymetry"
    else:
        raise ValueError("No bathymetry or depth variable found in dataset")

    depths = ds[depth_var].values
    lats = ds.lat_center.values
    lons = ds.lon_center.values

    # Create a line between the two points
    start_lat, start_lon = start_point
    end_lat, end_lon = end_point

    # Generate points along the line
    fractions = np.linspace(0, 1, num_points)
    lats_line = start_lat + fractions * (end_lat - start_lat)
    lons_line = start_lon + fractions * (end_lon - start_lon)

    # Calculate distances along the line
    distances = [0]
    for i in range(1, len(lats_line)):
        dist = haversine_distance(
            lats_line[i - 1], lons_line[i - 1], lats_line[i], lons_line[i]
        )
        distances.append(distances[-1] + dist)

    # Convert to numpy array and to kilometers
    distances = np.array(distances) / 1000  # Convert to km

    # Interpolate depths at the line points
    line_depths = griddata(
        (lats, lons), depths, (lats_line, lons_line), method="linear"
    )

    # If variable_name is provided, interpolate that as well
    if variable_name and variable_name in ds:
        var_data = ds[variable_name].values
        line_var_data = griddata(
            (lats, lons), var_data, (lats_line, lons_line), method="linear"
        )
    else:
        line_var_data = None

    # Create the figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Flip the depths to represent underwater correctly
    line_depths = -line_depths

    # Plot the depth profile
    ax.fill_between(distances, line_depths, 0, color="lightblue", alpha=0.5)
    ax.plot(distances, line_depths, color="blue", linewidth=2, label="Seafloor")

    # Plot zero line (sea level)
    ax.axhline(y=0, color="navy", linestyle="-", alpha=0.7)

    # Add text for "sea level"
    sea_level_text = ax.text(
        distances[-1],
        0.2,
        "Sea Level",
        color="navy",
        ha="right",
        va="bottom",
        fontweight="bold",
    )
    sea_level_text.set_path_effects(
        [PathEffects.withStroke(linewidth=3, foreground="white")]
    )

    # If variable data is available, plot on second axis
    if line_var_data is not None:
        ax2 = ax.twinx()
        ax2.plot(distances, line_var_data, "r-", linewidth=2, label=variable_name)
        ax2.set_ylabel(
            f"{variable_name} [{ds[variable_name].attrs.get('units', '')}]", color="r"
        )
        ax2.tick_params(axis="y", labelcolor="r")

        # Add second legend
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines2, labels2, loc="upper right")

    # Add labels and title
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Depth (m)")
    ax.set_title(
        f"Bathymetry Profile: ({start_lat:.4f}°N, {start_lon:.4f}°E) to ({end_lat:.4f}°N, {end_lon:.4f}°E)"
    )

    # Reverse y-axis to show depth going down
    ax.invert_yaxis()

    # Add grid
    ax.grid(True, linestyle="--", alpha=0.7)

    # Add legend for the depth profile
    ax.legend(loc="upper left")

    # Add start and end point annotations
    ax.annotate(
        "Start",
        xy=(distances[0], line_depths[0]),
        xytext=(distances[0], line_depths[0] - 5),
        arrowprops=dict(facecolor="black", shrink=0.05, width=1.5, headwidth=8),
        fontweight="bold",
    )

    ax.annotate(
        "End",
        xy=(distances[-1], line_depths[-1]),
        xytext=(distances[-1], line_depths[-1] - 5),
        arrowprops=dict(facecolor="black", shrink=0.05, width=1.5, headwidth=8),
        fontweight="bold",
    )

    plt.tight_layout()
    return fig


def add_profile_line_to_map(ax, start_point, end_point, transformer_to_web_mercator):
    """
    Add a profile line to an existing map, with distance markers.

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        Axes to add the line to
    start_point : tuple
        (lat, lon) of the starting point
    end_point : tuple
        (lat, lon) of the ending point
    transformer_to_web_mercator : callable
        Function to transform from lat/lon to web mercator coordinates

    Returns:
    --------
    dict
        Information about the profile line
    """
    start_lat, start_lon = start_point
    end_lat, end_lon = end_point

    # Transform to web mercator
    start_x, start_y = transformer_to_web_mercator.transform(start_lon, start_lat)
    end_x, end_y = transformer_to_web_mercator.transform(end_lon, end_lat)

    # Draw the line
    line = ax.plot([start_x, end_x], [start_y, end_y], "r-", linewidth=2, zorder=10)

    # Add markers at start and end
    ax.plot(start_x, start_y, "go", markersize=8, zorder=11)
    ax.plot(end_x, end_y, "ro", markersize=8, zorder=11)

    # Add text labels
    ax.text(
        start_x,
        start_y,
        "A",
        color="white",
        fontweight="bold",
        ha="center",
        va="center",
        zorder=12,
        bbox=dict(facecolor="green", alpha=0.7, boxstyle="circle"),
    )

    ax.text(
        end_x,
        end_y,
        "B",
        color="white",
        fontweight="bold",
        ha="center",
        va="center",
        zorder=12,
        bbox=dict(facecolor="red", alpha=0.7, boxstyle="circle"),
    )

    # Calculate profile distance
    distance = haversine_distance(start_lat, start_lon, end_lat, end_lon)

    # Add distance annotation at the middle of the line
    mid_x = (start_x + end_x) / 2
    mid_y = (start_y + end_y) / 2

    ax.text(
        mid_x,
        mid_y,
        f"{distance/1000:.1f} km",
        color="red",
        fontweight="bold",
        ha="center",
        va="center",
        zorder=12,
        bbox=dict(facecolor="white", alpha=0.7, boxstyle="round,pad=0.3"),
    )

    return {"start_point": start_point, "end_point": end_point, "distance_m": distance}


def create_cross_section_with_flow(
    ds, start_point, end_point, variable_name="speed", depth_levels=10, time_index=0
):
    """
    Create a cross-section visualization showing flow speeds at different depths.

    Parameters:
    -----------
    ds : xarray.Dataset
        Dataset containing flow data at multiple depths
    start_point : tuple
        (lat, lon) of the starting point
    end_point : tuple
        (lat, lon) of the ending point
    variable_name : str, optional
        Name of the flow variable to visualize
    depth_levels : int, optional
        Number of depth levels to sample
    time_index : int, optional
        Time index to use for time-varying data

    Returns:
    --------
    matplotlib.figure.Figure
        Figure containing the cross-section visualization
    """
    # Check if dataset has sigma layers
    has_sigma_layers = "sigma_layer" in ds.dims

    if not has_sigma_layers:
        raise ValueError(
            "Dataset does not have sigma layers for vertical cross-section"
        )

    # Extract bathymetry data
    if "seafloor_depth" in ds:
        depth_var = "seafloor_depth"
    elif "depth" in ds:
        depth_var = "depth"
    elif "bathymetry" in ds:
        depth_var = "bathymetry"
    else:
        raise ValueError("No bathymetry or depth variable found in dataset")

    depths = ds[depth_var].values
    lats = ds.lat_center.values
    lons = ds.lon_center.values

    # Create a line between the two points
    start_lat, start_lon = start_point
    end_lat, end_lon = end_point

    # Number of points along the profile
    num_points = 50

    # Generate points along the line
    fractions = np.linspace(0, 1, num_points)
    lats_line = start_lat + fractions * (end_lat - start_lat)
    lons_line = start_lon + fractions * (end_lon - start_lon)

    # Calculate distances along the line
    distances = [0]
    for i in range(1, len(lats_line)):
        dist = haversine_distance(
            lats_line[i - 1], lons_line[i - 1], lats_line[i], lons_line[i]
        )
        distances.append(distances[-1] + dist)

    # Convert to numpy array and to kilometers
    distances = np.array(distances) / 1000  # Convert to km

    # Interpolate depths at the line points
    line_depths = griddata(
        (lats, lons), depths, (lats_line, lons_line), method="linear"
    )

    # Create depth levels
    max_depth = np.max(line_depths)
    depth_values = np.linspace(0, max_depth, depth_levels)

    # Create a 2D grid: horizontal distance x depth
    X, Y = np.meshgrid(distances, depth_values)

    # Interpolate flow data at each depth level
    flow_data = np.zeros((depth_levels, num_points))

    # Get flow data at each sigma layer
    num_sigma_layers = ds.dims["sigma_layer"]

    if "time" in ds.dims:
        # Use specified time index
        ds_time = ds.isel(time=time_index)
    else:
        ds_time = ds

    # For each depth level
    for i, depth in enumerate(depth_values):
        # Find the closest sigma layer for this depth
        # This is a simplification - in reality, sigma layers are proportional to water depth
        sigma_idx = min(
            int(i * (num_sigma_layers / depth_levels)), num_sigma_layers - 1
        )

        # Get flow data at this sigma layer
        layer_data = ds_time[variable_name].isel(sigma_layer=sigma_idx).values

        # Interpolate onto our line
        layer_line_data = griddata(
            (lats, lons), layer_data, (lats_line, lons_line), method="linear"
        )

        # Store in our 2D array
        flow_data[i] = layer_line_data

    # Create the figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot the bathymetry profile
    ax.fill_between(distances, -line_depths, -max_depth, color="burlywood", alpha=0.7)
    ax.plot(distances, -line_depths, color="brown", linewidth=2)

    # Plot flow data as a contour
    contour_levels = 20
    cs = ax.contourf(X, -Y, flow_data, contour_levels, cmap="viridis", alpha=0.8)

    # Add colorbar
    cbar = plt.colorbar(cs, ax=ax)
    cbar.set_label(f"{variable_name} [{ds[variable_name].attrs.get('units', '')}]")

    # Plot zero line (sea level)
    ax.axhline(y=0, color="navy", linestyle="-", alpha=0.7)

    # Add text for "sea level"
    sea_level_text = ax.text(
        distances[-1],
        0.2,
        "Sea Level",
        color="navy",
        ha="right",
        va="bottom",
        fontweight="bold",
    )
    sea_level_text.set_path_effects(
        [PathEffects.withStroke(linewidth=3, foreground="white")]
    )

    # Add labels and title
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Depth (m)")
    ax.set_title(
        f"Flow Cross-Section: ({start_lat:.4f}°N, {start_lon:.4f}°E) to ({end_lat:.4f}°N, {end_lon:.4f}°E)"
    )

    # Set y-axis limits
    ax.set_ylim(-max_depth, 1)

    # Add grid
    ax.grid(True, linestyle="--", alpha=0.7)

    # Add start and end point annotations
    ax.annotate(
        "A",
        xy=(distances[0], -line_depths[0]),
        xytext=(distances[0], -line_depths[0] - 2),
        ha="center",
        fontweight="bold",
    )

    ax.annotate(
        "B",
        xy=(distances[-1], -line_depths[-1]),
        xytext=(distances[-1], -line_depths[-1] - 2),
        ha="center",
        fontweight="bold",
    )

    plt.tight_layout()
    return fig
