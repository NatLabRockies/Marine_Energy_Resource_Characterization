# time_series module - Part of tidal_visualizer package

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import Normalize
from datetime import datetime, timedelta

from .io import get_variable_data


def create_time_series_visualization(
    ds,
    variable_name,
    location_info,
    area_info=None,
    time_steps=24,
    output_file=None,
    fps=4,
):
    """
    Create an animated visualization showing how tidal conditions change over time.

    Parameters:
    -----------
    ds : xarray.Dataset
        Dataset containing time-varying tidal data
    variable_name : str
        Name of the variable to visualize
    location_info : dict
        Information about the location
    area_info : dict, optional
        Information about a specific area of interest
    time_steps : int, optional
        Number of time steps to include in the animation
    output_file : str, optional
        Path to save the animation file
    fps : int, optional
        Frames per second for the animation

    Returns:
    --------
    matplotlib.animation.Animation
        Animation object
    """
    from .visualization import TidalVisualizer

    # Create visualizer
    visualizer = TidalVisualizer({"locations": {"temp": {"hotspots": {}}}})

    # Get time dimension
    time_dim = None
    for dim in ds.dims:
        if dim.lower() in ["time", "t"]:
            time_dim = dim
            break

    if time_dim is None:
        raise ValueError("No time dimension found in dataset")

    # Determine time range
    all_times = ds[time_dim].values
    time_indices = np.linspace(0, len(all_times) - 1, time_steps, dtype=int)

    # Setup figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Get spatial information
    if area_info:
        bounds = area_info["bounds"]
        title = f"{location_info.get('location_display_name', '')} - {area_info.get('display_name', '')}"
    else:
        # Get bounds from all areas in the location
        bounds_list = [
            area["bounds"] for area in location_info.get("hotspots", {}).values()
        ]
        if not bounds_list:
            raise ValueError("No areas of interest found for location")

        bounds = visualizer.base.calculate_combined_bounds(bounds_list)
        title = f"{location_info.get('location_display_name', '')}"

    # Determine appropriate zoom level
    zoom_level = visualizer.base.determine_zoom_level_from_bounds(bounds)

    # Create mesh triangulation
    triang, _ = visualizer.create_mesh_triangulation(ds)

    # Set up normalization for consistent color scale
    all_values = []
    for idx in time_indices:
        time_data = get_variable_data(ds.isel({time_dim: idx}), variable_name)
        all_values.extend(time_data)

    vmin, vmax = np.min(all_values), np.max(all_values)
    norm = Normalize(vmin=vmin, vmax=vmax)

    # Function to update the plot for each frame
    def update(frame):
        ax.clear()

        # Get data for this time step
        time_idx = time_indices[frame]
        time_data = get_variable_data(ds.isel({time_dim: time_idx}), variable_name)
        time_value = all_times[time_idx]

        # Format timestamp
        if np.issubdtype(time_value.dtype, np.datetime64):
            time_str = pd.to_datetime(time_value).strftime("%Y-%m-%d %H:%M")
        else:
            time_str = f"Time step: {time_value}"

        # Plot the data
        tcf = ax.tripcolor(triang, time_data, cmap=visualizer.colormap, norm=norm)

        # Add basemap
        visualizer.add_basemap_to_visualization(ax, zoom_level)

        # Set bounds
        x_min, y_min = visualizer.base.transformer_to_web_mercator.transform(
            bounds[0], bounds[1]
        )
        x_max, y_max = visualizer.base.transformer_to_web_mercator.transform(
            bounds[2], bounds[3]
        )
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        # Add title with timestamp
        ax.set_title(f"{title}\n{variable_name} - {time_str}", fontsize=12)

        # Add north arrow and lat/lon ticks
        visualizer.add_north_arrow(ax)
        visualizer.add_latlon_ticks(ax)

        return [tcf]

    # Create the animation
    ani = animation.FuncAnimation(fig, update, frames=len(time_indices), blit=False)

    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=visualizer.colormap), cax=cbar_ax
    )
    cbar.set_label(variable_name)

    # Save if output file is provided
    if output_file:
        ani.save(output_file, writer="pillow", fps=fps)
        print(f"Animation saved to {output_file}")

    return ani


def plot_time_series_at_point(ds, variable_name, lat, lon, time_window=None):
    """
    Create a time series plot for a specific point in the dataset.

    Parameters:
    -----------
    ds : xarray.Dataset
        Dataset containing time-varying tidal data
    variable_name : str
        Name of the variable to visualize
    lat : float
        Latitude of the point
    lon : float
        Longitude of the point
    time_window : tuple, optional
        (start_time, end_time) to restrict the time range

    Returns:
    --------
    matplotlib.figure.Figure
        Figure containing the time series plot
    """
    # Find time dimension
    time_dim = None
    for dim in ds.dims:
        if dim.lower() in ["time", "t"]:
            time_dim = dim
            break

    if time_dim is None:
        raise ValueError("No time dimension found in dataset")

    # Find the nearest grid point to the specified coordinates
    lats = ds.lat_center.values
    lons = ds.lon_center.values

    # Calculate distances from the point to all grid points
    distances = np.sqrt((lats - lat) ** 2 + (lons - lon) ** 2)
    nearest_idx = np.argmin(distances)

    nearest_lat = lats[nearest_idx]
    nearest_lon = lons[nearest_idx]

    # Extract the time series
    # This will depend on the specific structure of your dataset
    # Here's an approach that works if the variable is organized with dimension like (time, cell)
    try:
        if "cell" in ds[variable_name].dims:
            time_series = ds[variable_name].isel(cell=nearest_idx).values
        else:
            # Try to extract based on the index pattern
            indices = np.where((lats == nearest_lat) & (lons == nearest_lon))[0]
            if len(indices) > 0:
                idx = indices[0]
                time_series = (
                    ds[variable_name].isel({ds[variable_name].dims[-1]: idx}).values
                )
            else:
                raise ValueError(
                    "Could not identify the correct indices for time series extraction"
                )
    except Exception as e:
        print(f"Error extracting time series: {e}")
        print("Attempting alternative extraction method...")

        # Try to extract the nearest point for each time step
        time_steps = ds[time_dim].size
        time_series = np.zeros(time_steps)

        for t in range(time_steps):
            time_data = get_variable_data(ds.isel({time_dim: t}), variable_name)
            time_series[t] = time_data[nearest_idx]

    # Get time values
    times = ds[time_dim].values

    # Apply time window if specified
    if time_window:
        start_time, end_time = time_window
        mask = (times >= start_time) & (times <= end_time)
        times = times[mask]
        time_series = time_series[mask]

    # Create the figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot the time series
    ax.plot(times, time_series, marker="o", linestyle="-", markersize=4)

    # Add labels and title
    ax.set_xlabel("Time")
    ax.set_ylabel(f"{variable_name} [{ds[variable_name].attrs.get('units', '')}]")
    ax.set_title(f"Time Series at ({nearest_lat:.4f}°N, {nearest_lon:.4f}°E)")

    # Format x-axis if times are datetime64
    if np.issubdtype(times.dtype, np.datetime64):
        import matplotlib.dates as mdates

        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
        plt.xticks(rotation=45)

    # Add grid
    ax.grid(True, linestyle="--", alpha=0.7)

    # Add annotation about the exact location
    ax.annotate(
        f"Nearest grid point: ({nearest_lat:.4f}°N, {nearest_lon:.4f}°E)",
        xy=(0.05, 0.95),
        xycoords="axes fraction",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
    )

    plt.tight_layout()
    return fig
