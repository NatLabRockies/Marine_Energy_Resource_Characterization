import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import matplotlib.cm as cm


def add_flow_vectors(
    ax,
    ds,
    scale=0.01,
    density=10,
    color="white",
    width=0.5,
    transform_to_web_mercator=True,
    transformer_to_web_mercator=None,
):
    """
    Add flow velocity vectors to a plot.

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        Axes to add the vectors to
    ds : xarray.Dataset
        Dataset containing velocity components
    scale : float, optional
        Scaling factor for vectors
    density : int, optional
        Sampling density (higher = fewer vectors)
    color : str or callable, optional
        Color for vectors or function to determine color
    width : float, optional
        Line width for vectors
    transform_to_web_mercator : bool, optional
        Whether to transform coordinates to web mercator
    transformer_to_web_mercator : callable, optional
        Function to transform from lat/lon to web mercator coordinates

    Returns:
    --------
    matplotlib.quiver.Quiver
        The quiver plot object
    """
    # Identify the velocity component variables
    u_var = None
    v_var = None

    # Common names for eastward and northward velocity components
    u_candidates = [
        "u",
        "u_velocity",
        "eastward_velocity",
        "u_component",
        "east_velocity",
    ]
    v_candidates = [
        "v",
        "v_velocity",
        "northward_velocity",
        "v_component",
        "north_velocity",
    ]

    # Try to find u component
    for var in u_candidates:
        if var in ds:
            u_var = var
            break

    # Try to find v component
    for var in v_candidates:
        if var in ds:
            v_var = var
            break

    if u_var is None or v_var is None:
        raise ValueError("Could not identify velocity component variables")

    # Extract coordinates and velocity components
    lats = ds.lat_center.values
    lons = ds.lon_center.values
    u = ds[u_var].values
    v = ds[v_var].values

    # Sample the data for cleaner visualization
    # Choose every Nth point
    idx = slice(None, None, density)
    lats_sampled = lats[idx]
    lons_sampled = lons[idx]
    u_sampled = u[idx]
    v_sampled = v[idx]

    # Transform coordinates if requested
    if transform_to_web_mercator and transformer_to_web_mercator is not None:
        x, y = transformer_to_web_mercator.transform(lons_sampled, lats_sampled)
    else:
        x, y = lons_sampled, lats_sampled

    # Create quiver plot
    quiver = ax.quiver(
        x,
        y,
        u_sampled,
        v_sampled,
        scale=1.0 / scale,
        color=color,
        width=width,
        headwidth=3,
        headlength=4,
        headaxislength=3.5,
        zorder=15,
    )

    return quiver


def add_scaled_flow_vectors(
    ax,
    ds,
    scale_var="speed",
    cmap="viridis",
    norm=None,
    min_length=0.1,
    max_length=10.0,
    width_scale=1.0,
    density=10,
    transform_to_web_mercator=True,
    transformer_to_web_mercator=None,
):
    """
    Add flow velocity vectors scaled by speed or another variable.

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        Axes to add the vectors to
    ds : xarray.Dataset
        Dataset containing velocity components
    scale_var : str, optional
        Variable to use for scaling vectors
    cmap : str or matplotlib.colors.Colormap, optional
        Colormap for vectors
    norm : matplotlib.colors.Normalize, optional
        Normalization for color mapping
    min_length : float, optional
        Minimum arrow length
    max_length : float, optional
        Maximum arrow length
    width_scale : float, optional
        Scaling factor for arrow width
    density : int, optional
        Sampling density (higher = fewer vectors)
    transform_to_web_mercator : bool, optional
        Whether to transform coordinates to web mercator
    transformer_to_web_mercator : callable, optional
        Function to transform from lat/lon to web mercator coordinates

    Returns:
    --------
    matplotlib.quiver.Quiver
        The quiver plot object
    """
    # Identify the velocity component variables
    u_var = None
    v_var = None

    # Common names for eastward and northward velocity components
    u_candidates = [
        "u",
        "u_velocity",
        "eastward_velocity",
        "u_component",
        "east_velocity",
    ]
    v_candidates = [
        "v",
        "v_velocity",
        "northward_velocity",
        "v_component",
        "north_velocity",
    ]

    # Try to find u component
    for var in u_candidates:
        if var in ds:
            u_var = var
            break

    # Try to find v component
    for var in v_candidates:
        if var in ds:
            v_var = var
            break

    if u_var is None or v_var is None:
        raise ValueError("Could not identify velocity component variables")

    # Extract coordinates and velocity components
    lats = ds.lat_center.values
    lons = ds.lon_center.values
    u = ds[u_var].values
    v = ds[v_var].values

    # Get the scaling variable
    if scale_var in ds:
        scale_data = ds[scale_var].values
    elif scale_var == "speed" and "speed" not in ds:
        # Calculate speed if not available
        scale_data = np.sqrt(u**2 + v**2)
    else:
        raise ValueError(f"Scale variable '{scale_var}' not found in dataset")

    # Sample the data for cleaner visualization
    # Choose every Nth point
    idx = slice(None, None, density)
    lats_sampled = lats[idx]
    lons_sampled = lons[idx]
    u_sampled = u[idx]
    v_sampled = v[idx]
    scale_data_sampled = scale_data[idx]

    # Transform coordinates if requested
    if transform_to_web_mercator and transformer_to_web_mercator is not None:
        x, y = transformer_to_web_mercator.transform(lons_sampled, lats_sampled)
    else:
        x, y = lons_sampled, lats_sampled

    # Create normalization if not provided
    if norm is None:
        norm = Normalize(
            vmin=np.min(scale_data_sampled), vmax=np.max(scale_data_sampled)
        )

    # Convert colormap string to colormap object
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)

    # Calculate arrow colors
    colors = cmap(norm(scale_data_sampled))

    # Calculate arrow lengths (scaling between min and max length)
    normalized_lengths = norm(scale_data_sampled)
    arrow_lengths = min_length + (max_length - min_length) * normalized_lengths

    # Calculate arrow widths (scaling with length)
    arrow_widths = width_scale * (0.001 + 0.004 * normalized_lengths)

    # Create quiver plot with scaled arrows and colors
    quiver = ax.quiver(
        x,
        y,
        u_sampled,
        v_sampled,
        color=colors,
        scale=None,  # Let arrow_lengths determine the scale
        scale_units="inches",
        width=arrow_widths,
        headwidth=3,
        headlength=5,
        headaxislength=4.5,
        zorder=15,
    )

    return quiver


def add_streamlines(
    ax,
    ds,
    density=1.0,
    color="white",
    linewidth=1.0,
    arrowsize=1.0,
    transform_to_web_mercator=True,
    transformer_to_web_mercator=None,
):
    """
    Add streamlines to visualize flow patterns.

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        Axes to add the streamlines to
    ds : xarray.Dataset
        Dataset containing velocity components
    density : float or tuple, optional
        Density of streamlines
    color : str or callable, optional
        Color for streamlines
    linewidth : float or callable, optional
        Line width for streamlines
    arrowsize : float, optional
        Size of arrowheads
    transform_to_web_mercator : bool, optional
        Whether to transform coordinates to web mercator
    transformer_to_web_mercator : callable, optional
        Function to transform from lat/lon to web mercator coordinates

    Returns:
    --------
    matplotlib.streamplot.StreamplotSet
        The streamplot object
    """
    # Identify the velocity component variables
    u_var = None
    v_var = None

    # Common names for eastward and northward velocity components
    u_candidates = [
        "u",
        "u_velocity",
        "eastward_velocity",
        "u_component",
        "east_velocity",
    ]
    v_candidates = [
        "v",
        "v_velocity",
        "northward_velocity",
        "v_component",
        "north_velocity",
    ]

    # Try to find u component
    for var in u_candidates:
        if var in ds:
            u_var = var
            break

    # Try to find v component
    for var in v_candidates:
        if var in ds:
            v_var = var
            break

    if u_var is None or v_var is None:
        raise ValueError("Could not identify velocity component variables")

    # Extract coordinates and velocity components
    lats = ds.lat_center.values
    lons = ds.lon_center.values
    u = ds[u_var].values
    v = ds[v_var].values

    # For streamlines, we need data on a regular grid
    # Check if the data is already on a regular grid
    lon_unique = np.unique(lons)
    lat_unique = np.unique(lats)

    if len(lon_unique) * len(lat_unique) == len(lons):
        # Data is on a regular grid, reshape
        lon_grid, lat_grid = np.meshgrid(lon_unique, lat_unique)
        u_grid = u.reshape(len(lat_unique), len(lon_unique))
        v_grid = v.reshape(len(lat_unique), len(lon_unique))
    else:
        # Data is not on a regular grid, interpolate
        # Create a regular grid for interpolation
        lon_min, lon_max = np.min(lons), np.max(lons)
        lat_min, lat_max = np.min(lats), np.max(lats)

        grid_size = int(np.sqrt(len(lons)))  # Use a reasonable grid size

        lon_grid, lat_grid = np.meshgrid(
            np.linspace(lon_min, lon_max, grid_size),
            np.linspace(lat_min, lat_max, grid_size),
        )

        # Use griddata for interpolation
        from scipy.interpolate import griddata

        u_grid = griddata((lons, lats), u, (lon_grid, lat_grid), method="linear")
        v_grid = griddata((lons, lats), v, (lon_grid, lat_grid), method="linear")

    # Transform coordinates if requested
    if transform_to_web_mercator and transformer_to_web_mercator is not None:
        x_grid, y_grid = transformer_to_web_mercator.transform(lon_grid, lat_grid)
    else:
        x_grid, y_grid = lon_grid, lat_grid

    # Create streamlines
    streamlines = ax.streamplot(
        x_grid,
        y_grid,
        u_grid,
        v_grid,
        density=density,
        color=color,
        linewidth=linewidth,
        arrowsize=arrowsize,
        arrowstyle="-|>",
        zorder=15,
    )

    return streamlines


def create_composite_visualization(
    ax,
    ds,
    background_var,
    overlay_var=None,
    vector_var=None,
    background_cmap="viridis",
    overlay_cmap="plasma",
    background_norm=None,
    overlay_norm=None,
    overlay_alpha=0.5,
    vector_color="white",
    vector_density=20,
    vector_scale=0.01,
    transform_to_web_mercator=True,
    transformer_to_web_mercator=None,
    use_streamlines=False,
):
    """
    Create a composite visualization with multiple variables.

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        Axes to create the visualization on
    ds : xarray.Dataset
        Dataset containing the variables
    background_var : str
        Variable to use for the background
    overlay_var : str, optional
        Variable to overlay on the background
    vector_var : str, optional
        Variable to use for vector field visualization
    background_cmap : str or matplotlib.colors.Colormap, optional
        Colormap for the background
    overlay_cmap : str or matplotlib.colors.Colormap, optional
        Colormap for the overlay
    background_norm : matplotlib.colors.Normalize, optional
        Normalization for the background
    overlay_norm : matplotlib.colors.Normalize, optional
        Normalization for the overlay
    overlay_alpha : float, optional
        Alpha (transparency) for the overlay
    vector_color : str or callable, optional
        Color for the vector field
    vector_density : int, optional
        Density of vectors or streamlines
    vector_scale : float, optional
        Scaling factor for vectors
    transform_to_web_mercator : bool, optional
        Whether to transform coordinates to web mercator
    transformer_to_web_mercator : callable, optional
        Function to transform from lat/lon to web mercator coordinates
    use_streamlines : bool, optional
        Whether to use streamlines instead of arrows for vector field

    Returns:
    --------
    dict
        Dictionary of visualization components
    """
    # Dictionary to store visualization components
    components = {}

    # Extract coordinates
    lats = ds.lat_center.values
    lons = ds.lon_center.values

    # Transform coordinates if requested
    if transform_to_web_mercator and transformer_to_web_mercator is not None:
        x, y = transformer_to_web_mercator.transform(lons, lats)
    else:
        x, y = lons, lats

    # Plot background
    if background_var in ds:
        background_data = ds[background_var].values
        background_triang = plt.matplotlib.tri.Triangulation(x, y)

        # Create normalization if not provided
        if background_norm is None:
            background_norm = Normalize(
                vmin=np.nanmin(background_data), vmax=np.nanmax(background_data)
            )

        # Plot background
        background = ax.tripcolor(
            background_triang,
            background_data,
            cmap=background_cmap,
            norm=background_norm,
            shading="gouraud",
            zorder=5,
        )

        components["background"] = background

        # Add colorbar for background
        background_cbar = plt.colorbar(
            background,
            ax=ax,
            orientation="vertical",
            pad=0.02,
            shrink=0.8,
            extend="both",
        )
        background_cbar.set_label(background_var)
        components["background_cbar"] = background_cbar

    # Plot overlay if provided
    if overlay_var is not None and overlay_var in ds:
        overlay_data = ds[overlay_var].values
        overlay_triang = plt.matplotlib.tri.Triangulation(x, y)

        # Create normalization if not provided
        if overlay_norm is None:
            overlay_norm = Normalize(
                vmin=np.nanmin(overlay_data), vmax=np.nanmax(overlay_data)
            )

        # Plot overlay
        overlay = ax.tripcolor(
            overlay_triang,
            overlay_data,
            cmap=overlay_cmap,
            norm=overlay_norm,
            alpha=overlay_alpha,
            shading="gouraud",
            zorder=10,
        )

        components["overlay"] = overlay

        # Add colorbar for overlay
        overlay_cbar = plt.colorbar(
            overlay, ax=ax, orientation="horizontal", pad=0.1, shrink=0.8, extend="both"
        )
        overlay_cbar.set_label(overlay_var)
        components["overlay_cbar"] = overlay_cbar

    # Add vector field if requested
    if vector_var is not None:
        if use_streamlines:
            # Add streamlines
            streamlines = add_streamlines(
                ax,
                ds,
                density=1.0 / vector_density,
                color=vector_color,
                linewidth=0.8,
                arrowsize=1.2,
                transform_to_web_mercator=transform_to_web_mercator,
                transformer_to_web_mercator=transformer_to_web_mercator,
            )
            components["vector_field"] = streamlines
        else:
            # Add arrows
            quiver = add_flow_vectors(
                ax,
                ds,
                scale=vector_scale,
                density=vector_density,
                color=vector_color,
                width=0.5,
                transform_to_web_mercator=transform_to_web_mercator,
                transformer_to_web_mercator=transformer_to_web_mercator,
            )
            components["vector_field"] = quiver

            # Add key for scale reference
            key = ax.quiverkey(
                quiver,
                0.9,
                0.95,
                1.0,
                "1 m/s",
                labelpos="E",
                coordinates="figure",
                fontproperties={"size": 10},
            )
            components["vector_key"] = key

    return components


def create_particle_animation(
    ax,
    ds,
    time_dim="time",
    frames=20,
    density=50,
    cmap="viridis",
    norm=None,
    size=2,
    alpha=0.7,
    transform_to_web_mercator=True,
    transformer_to_web_mercator=None,
):
    """
    Create a particle animation to visualize flow over time.

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        Axes to create the animation on
    ds : xarray.Dataset
        Dataset containing velocity components over time
    time_dim : str, optional
        Name of the time dimension
    frames : int, optional
        Number of frames for the animation
    density : int, optional
        Number of particles
    cmap : str or matplotlib.colors.Colormap, optional
        Colormap for particles
    norm : matplotlib.colors.Normalize, optional
        Normalization for particle colors
    size : float, optional
        Size of particles
    alpha : float, optional
        Alpha (transparency) for particles
    transform_to_web_mercator : bool, optional
        Whether to transform coordinates to web mercator
    transformer_to_web_mercator : callable, optional
        Function to transform from lat/lon to web mercator coordinates

    Returns:
    --------
    matplotlib.animation.FuncAnimation
        The animation object
    """
    import matplotlib.animation as animation

    # Identify the velocity component variables
    u_var = None
    v_var = None

    # Common names for eastward and northward velocity components
    u_candidates = [
        "u",
        "u_velocity",
        "eastward_velocity",
        "u_component",
        "east_velocity",
    ]
    v_candidates = [
        "v",
        "v_velocity",
        "northward_velocity",
        "v_component",
        "north_velocity",
    ]

    # Try to find u component
    for var in u_candidates:
        if var in ds:
            u_var = var
            break

    # Try to find v component
    for var in v_candidates:
        if var in ds:
            v_var = var
            break

    if u_var is None or v_var is None:
        raise ValueError("Could not identify velocity component variables")

    # Check if time dimension exists
    if time_dim not in ds.dims:
        raise ValueError(f"Time dimension '{time_dim}' not found in dataset")

    # Extract spatial coordinates
    lats = ds.lat_center.values
    lons = ds.lon_center.values

    # Get time values
    times = ds[time_dim].values

    # Sample time steps for animation
    time_indices = np.linspace(0, len(times) - 1, frames, dtype=int)

    # Transform coordinates if requested
    if transform_to_web_mercator and transformer_to_web_mercator is not None:
        x, y = transformer_to_web_mercator.transform(lons, lats)
    else:
        x, y = lons, lats

    # Initialize random particle positions within the domain
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)

    # Create particles
    particles_x = np.random.uniform(x_min, x_max, density)
    particles_y = np.random.uniform(y_min, y_max, density)

    # Get speed field for the first time step to use for particle colors
    u_initial = ds[u_var].isel({time_dim: 0}).values
    v_initial = ds[v_var].isel({time_dim: 0}).values
    speed_initial = np.sqrt(u_initial**2 + v_initial**2)

    # Create normalization for particle colors if not provided
    if norm is None:
        norm = Normalize(vmin=np.min(speed_initial), vmax=np.max(speed_initial))

    # Convert colormap string to colormap object
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)

    # Initialize the scatter plot for particles
    scatter = ax.scatter(
        particles_x,
        particles_y,
        c="white",  # Initial color, will be updated in animation
        s=size,
        alpha=alpha,
        zorder=20,
    )

    # Function to interpolate velocity at particle positions
    def interpolate_velocity(x_pos, y_pos, u_field, v_field):
        from scipy.interpolate import griddata

        # Interpolate u and v at particle positions
        u_interp = griddata(
            (x, y), u_field, (x_pos, y_pos), method="linear", fill_value=0
        )
        v_interp = griddata(
            (x, y), v_field, (x_pos, y_pos), method="linear", fill_value=0
        )

        return u_interp, v_interp

    # Preprocessing for interpolation if the grid is regular
    grid_interpolator = None
    lon_unique = np.unique(lons)
    lat_unique = np.unique(lats)

    if len(lon_unique) * len(lat_unique) == len(lons):
        # Data is on a regular grid, can use faster RegularGridInterpolator
        from scipy.interpolate import RegularGridInterpolator

        # Reshape initial u and v fields to 2D
        u_grid = u_initial.reshape(len(lat_unique), len(lon_unique))
        v_grid = v_initial.reshape(len(lat_unique), len(lon_unique))

        # Create interpolators
        grid_interpolator = {
            "u": RegularGridInterpolator(
                (lat_unique, lon_unique), u_grid, bounds_error=False, fill_value=0
            ),
            "v": RegularGridInterpolator(
                (lat_unique, lon_unique), v_grid, bounds_error=False, fill_value=0
            ),
        }

    # Animation update function
    def update(frame):
        nonlocal particles_x, particles_y

        # Get time index for this frame
        t_idx = time_indices[frame % len(time_indices)]

        # Get velocity field for this time
        u_field = ds[u_var].isel({time_dim: t_idx}).values
        v_field = ds[v_var].isel({time_dim: t_idx}).values

        # Interpolate velocity at particle positions
        if grid_interpolator is not None:
            # Convert particle positions to geographic coordinates
            if transform_to_web_mercator and transformer_to_web_mercator is not None:
                geo_x, geo_y = transformer_from_web_mercator.transform(
                    particles_x, particles_y
                )
            else:
                geo_x, geo_y = particles_x, particles_y

            # Interpolate using the RegularGridInterpolator
            points = np.column_stack([geo_y, geo_x])  # Note: lat, lon order
            u_interp = grid_interpolator["u"](points)
            v_interp = grid_interpolator["v"](points)
        else:
            # Use griddata for unstructured grids
            u_interp, v_interp = interpolate_velocity(
                particles_x, particles_y, u_field, v_field
            )

        # Calculate timestep based on maximum velocity
        # (this is a simplification - a real model would use adaptive timestepping)
        speed = np.sqrt(u_interp**2 + v_interp**2)
        max_speed = np.max(speed) if len(speed) > 0 else 1.0
        dt = 0.1 / max(max_speed, 1e-6)  # Avoid division by zero

        # Update particle positions
        particles_x += u_interp * dt
        particles_y += v_interp * dt

        # Reset particles that go out of bounds
        out_of_bounds = (
            (particles_x < x_min)
            | (particles_x > x_max)
            | (particles_y < y_min)
            | (particles_y > y_max)
        )

        # Replace out-of-bounds particles with new random positions
        num_reset = np.sum(out_of_bounds)
        if num_reset > 0:
            particles_x[out_of_bounds] = np.random.uniform(x_min, x_max, num_reset)
            particles_y[out_of_bounds] = np.random.uniform(y_min, y_max, num_reset)

        # Update particle colors based on speed
        colors = cmap(norm(speed))

        # Update the scatter plot
        scatter.set_offsets(np.column_stack([particles_x, particles_y]))
        scatter.set_color(colors)

        # Update title with current time
        if hasattr(times[t_idx], "strftime"):
            time_str = times[t_idx].strftime("%Y-%m-%d %H:%M")
        else:
            time_str = f"Time step: {times[t_idx]}"

        ax.set_title(f"Flow Particles Animation - {time_str}")

        return (scatter,)

    # Create animation
    anim = animation.FuncAnimation(
        ax.figure, update, frames=frames, interval=200, blit=True
    )

    return anim
