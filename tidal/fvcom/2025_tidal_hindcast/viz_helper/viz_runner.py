"""
Enhanced example script showing how to use the tidal_visualizer package.
This demonstrates all the advanced visualization features.
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import xarray as xr
import pandas as pd
import seaborn as sns

from tidal_visualizer import (
    # Core components
    TidalVisualizer,
    get_location_names,
    get_areas_of_interest,
    get_location_info,
    get_area_of_interest_info,
    find_dataset_file,
    load_dataset,
    # Enhanced visualization
    create_time_series_visualization,
    create_elevation_profile,
    add_profile_line_to_map,
    create_cross_section_with_flow,
    ContextLayerManager,
    CoordinateManager,
    add_flow_vectors,
    add_streamlines,
    create_composite_visualization,
)

# Sample configuration
viz_config = {
    "dirs": {
        # "base": "/Users/asimms/Desktop/Programming/resource_char_dev/marine_energy_resource_characterization/tidal/fvcom/2025_tidal_hindcast/data",
        "base": "/scratch/asimms/Tidal",
        "end": "b2_summary",
        # "output": "/Users/asimms/Desktop/Programming/resource_char_dev/marine_energy_resource_characterization/tidal/fvcom/2025_tidal_hindcast/viz_output",
        "output": "/scratch/asimms/Tidal/viz",
    },
    "variables": {
        "speed": {"min": 0, "max": 3},
        # "power_density": {"min": 0, "max": 10000},
        # "volume_energy_flux": {"min": 0, "max": 10000},
    },
    "locations": {
        "AK_cook_inlet": {
            "location_display_name": "Cook Inlet",
            "state": "Alaska",
            "state_abbr": "AK",
            "hotspots": {
                "east_forelands": {
                    "bounds": [-151.85, 60.65, -151.2, 60.8],
                    "display_name": "East Forelands",
                },
                "west_forelands": {
                    "bounds": [-152.15, 60.65, -151.85, 60.8],
                    "display_name": "West Forelands",
                },
            },
        },
        # "ME_western_passage": {
        #     "location_display_name": "Western Passage",
        #     "state": "Maine",
        #     "state_abbr": "ME",
        #     "hotspots": {
        #         "head_harbour_passage": {
        #             "bounds": [-66.95, 44.95, -66.75, 45.05],
        #             "display_name": "Head Harbour Passage",
        #         },
        #         "western_passage_main": {
        #             "bounds": [-67.1, 44.9, -66.95, 45.05],
        #             "display_name": "Western Passage",
        #         },
        #     },
        # },
    },
}


def create_output_dir():
    """Create output directory for examples."""
    output_dir = "example_outputs"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def example_enhanced_area_visualization(visualizer, location_key, area_key):
    """Example of enhanced area visualization with custom features."""
    print("\nCreating enhanced area visualization...")

    # Get location and area information
    location_info = get_location_info(viz_config, location_key)
    area_info = get_area_of_interest_info(viz_config, location_key, area_key)

    # Find and load dataset
    dataset_path = find_dataset_file(
        visualizer.base.base_data_dir,
        location_key,
        viz_config["dirs"].get("end", ""),
    )
    ds = load_dataset(dataset_path)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # Setup visualization context managers
    context_manager = ContextLayerManager(
        visualizer.base.transformer_to_web_mercator,
        visualizer.base.transformer_from_web_mercator,
    )

    # Add basic visualization
    bounds = area_info["bounds"]
    zoom_level = visualizer.base.determine_zoom_level_from_bounds(bounds)

    # Transform bounds to web mercator
    x_min, y_min = visualizer.base.transformer_to_web_mercator.transform(
        bounds[0], bounds[1]
    )
    x_max, y_max = visualizer.base.transformer_to_web_mercator.transform(
        bounds[2], bounds[3]
    )

    # Set plot bounds
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # Add ocean basemap
    context_manager.add_basemap(ax, basemap="ocean", zoom=zoom_level)

    # Add bathymetry contours
    context_manager.add_bathymetry_contours(
        ax,
        ds,
        contour_levels=np.arange(0, 200, 20),
        colors="blue",
        alpha=0.7,
        labels=True,
    )

    # Add flow visualization with composite view
    components = create_composite_visualization(
        ax,
        ds,
        background_var="power_density",
        vector_var="flow",
        background_cmap="viridis",
        vector_color="white",
        vector_density=15,
        vector_scale=0.02,
        transform_to_web_mercator=True,
        transformer_to_web_mercator=visualizer.base.transformer_to_web_mercator,
        use_streamlines=True,
    )

    # Add enhanced coordinate grid
    visualizer.add_coordinate_grid(ax, grid_spacing_degrees=0.1, minor=True)

    # Add multiple scale bars for different latitudes
    visualizer.add_multiple_scale_bars(ax)

    # Add profile line
    profile_start = (bounds[1] + 0.03, bounds[0] + 0.03)  # near bottom-left
    profile_end = (bounds[3] - 0.03, bounds[2] - 0.03)  # near top-right

    # profile_info = add_profile_line_to_map(
    #     ax, profile_start, profile_end, visualizer.base.transformer_to_web_mercator
    # )

    # Add an enhanced colorbar
    if "background" in components:
        visualizer.add_enhanced_colorbar(
            fig,
            ax,
            components["background"],
            "Power Density (W/m²)",
            position="vertical",
            extend="max",
        )

    # Add improved legend with custom items
    from matplotlib.lines import Line2D

    legend_items = [
        (Line2D([0], [0], color="blue", lw=1, linestyle="-"), "Bathymetry"),
        (Line2D([0], [0], color="white", lw=1.5), "Flow Streamlines"),
        (Line2D([0], [0], color="red", lw=2), "Profile Line"),
    ]

    visualizer.add_legend_with_custom_items(
        ax, legend_items, title="Map Elements", loc="upper left"
    )

    # Add title
    plt.title(
        f"{location_info['location_display_name']} - {area_info['display_name']}\nEnhanced Visualization",
        fontsize=14,
    )

    # Save figure
    output_dir = create_output_dir()
    output_path = os.path.join(output_dir, f"enhanced_{location_key}_{area_key}.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Enhanced visualization saved to {output_path}")

    # Create elevation profile
    profile_fig = create_elevation_profile(
        ds, profile_start, profile_end, variable_name="speed"
    )

    profile_output_path = os.path.join(
        output_dir, f"profile_{location_key}_{area_key}.png"
    )
    profile_fig.savefig(profile_output_path, dpi=300, bbox_inches="tight")
    print(f"Elevation profile saved to {profile_output_path}")

    # Create cross-section with flow
    try:
        cross_section_fig = create_cross_section_with_flow(
            ds, profile_start, profile_end, variable_name="speed", depth_levels=10
        )

        cross_section_output_path = os.path.join(
            output_dir, f"cross_section_{location_key}_{area_key}.png"
        )
        cross_section_fig.savefig(
            cross_section_output_path, dpi=300, bbox_inches="tight"
        )
        print(f"Cross-section visualization saved to {cross_section_output_path}")
    except Exception as e:
        print(f"Note: Could not create cross-section (this requires sigma layers): {e}")

    return fig, ax


def example_time_series(visualizer, location_key, area_key):
    """Example of time series visualization."""
    print("\nCreating time series visualization...")

    # This example requires time-varying data, which we'll simulate
    import xarray as xr
    from datetime import datetime, timedelta

    # Get area information
    area_info = get_area_of_interest_info(viz_config, location_key, area_key)
    location_info = get_location_info(viz_config, location_key)

    # Create a simple demo dataset with time dimension
    bounds = area_info["bounds"]
    lon_min, lat_min, lon_max, lat_max = bounds

    # Create grid
    grid_size = 30
    lons = np.linspace(lon_min, lon_max, grid_size)
    lats = np.linspace(lat_min, lat_max, grid_size)
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    # Flatten for unstructured grid
    lons_flat = lon_grid.flatten()
    lats_flat = lat_grid.flatten()

    # Create time dimension (12 hours with hourly data)
    times = [datetime(2023, 1, 1) + timedelta(hours=i) for i in range(12)]

    # Center point for flow patterns
    center_lon = (lon_min + lon_max) / 2
    center_lat = (lat_min + lat_max) / 2

    # Create time-varying data
    time_data = []
    for t, time in enumerate(times):
        # Create dummy data that varies with time
        # Phase of tidal cycle (0 to 2pi over 12 hours)
        phase = t * 2 * np.pi / 12

        # Distance from center
        dist_from_center = np.sqrt(
            (lons_flat - center_lon) ** 2 + (lats_flat - center_lat) ** 2
        )
        max_dist = np.max(dist_from_center)

        # Flow speed varies with time
        speed_factor = 0.5 + 0.5 * np.sin(phase)  # 0.5 to 1.5
        speed = 3 * speed_factor * (1 - dist_from_center / max_dist)

        # Flow directions (rotating with time)
        direction_offset = phase
        direction = (
            np.arctan2(lats_flat - center_lat, lons_flat - center_lon)
            + direction_offset
        )
        u = -np.sin(direction) * speed  # Eastward velocity
        v = np.cos(direction) * speed  # Northward velocity

        # Power density
        power_density = 0.5 * 1025 * speed**3  # rho * speed^3 / 2

        time_data.append(
            {"speed": speed, "u": u, "v": v, "power_density": power_density}
        )

    # Create dataset
    data_vars = {}
    for var_name in ["speed", "u", "v", "power_density"]:
        data_vars[var_name] = (
            ["time", "cell"],
            np.array([d[var_name] for d in time_data]),
        )

    # Add bathymetry (constant over time)
    dist_from_center = np.sqrt(
        (lons_flat - center_lon) ** 2 + (lats_flat - center_lat) ** 2
    )
    max_dist = np.max(dist_from_center)
    seafloor_depth = 50 + 100 * (1 - dist_from_center / max_dist) ** 2
    data_vars["seafloor_depth"] = (["cell"], seafloor_depth)

    # Create triangulation (dummy)
    dummy_triangulation = np.array(
        [np.arange(i, i + 3) % len(lats_flat) for i in range(len(lats_flat))]
    )

    # Create xarray dataset
    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            "time": times,
            "lat_center": (["cell"], lats_flat),
            "lon_center": (["cell"], lons_flat),
            "lat_node": (["node"], lats_flat),  # For triangulation
            "lon_node": (["node"], lons_flat),  # For triangulation
            "nv": (["nele", "three"], dummy_triangulation + 1),  # Dummy triangulation
        },
    )

    # Create time series visualization
    fig, ax = plt.subplots(figsize=(12, 10))

    try:
        # Create time series animation
        animation = create_time_series_visualization(
            ds, "speed", location_info, area_info, time_steps=12
        )

        # Save animation
        output_dir = create_output_dir()
        output_path = os.path.join(
            output_dir, f"animation_{location_key}_{area_key}.gif"
        )
        animation.save(output_path, writer="pillow", fps=2)
        print(f"Time series animation saved to {output_path}")
    except Exception as e:
        print(f"Note: Could not create animation (requires pillow): {e}")

        # As a fallback, create a static visualization
        # Setup visualization
        bounds = area_info["bounds"]
        zoom_level = visualizer.base.determine_zoom_level_from_bounds(bounds)

        # Transform bounds to web mercator
        x_min, y_min = visualizer.base.transformer_to_web_mercator.transform(
            bounds[0], bounds[1]
        )
        x_max, y_max = visualizer.base.transformer_to_web_mercator.transform(
            bounds[2], bounds[3]
        )

        # Set plot bounds
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        # Add basemap
        visualizer.add_basemap_to_visualization(ax, zoom_level)

        # Plot speed at first time step
        speed_data = ds["speed"].isel(time=0).values
        lats = ds.lat_center.values
        lons = ds.lon_center.values

        # Transform to web mercator
        x, y = visualizer.base.transformer_to_web_mercator.transform(lons, lats)

        # Create triangulation
        triang = plt.matplotlib.tri.Triangulation(x, y)

        # Plot speed
        tcf = ax.tripcolor(
            triang, speed_data, cmap="viridis", shading="gouraud", zorder=5
        )

        # Add vectors
        add_flow_vectors(
            ax,
            ds.isel(time=0),
            scale=0.02,
            density=10,
            color="white",
            transform_to_web_mercator=True,
            transformer_to_web_mercator=visualizer.base.transformer_to_web_mercator,
        )

        # Add colorbar
        cbar = plt.colorbar(tcf, ax=ax)
        cbar.set_label("Speed (m/s)")

        # Add title
        plt.title(
            f"{location_info['location_display_name']} - {area_info['display_name']}\nFlow Velocity (Static)",
            fontsize=14,
        )

        # Save figure
        output_dir = create_output_dir()
        output_path = os.path.join(
            output_dir, f"flow_static_{location_key}_{area_key}.png"
        )
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Static flow visualization saved to {output_path}")

    # Create point time series
    # point_lat = (bounds[1] + bounds[3]) / 2
    # point_lon = (bounds[0] + bounds[2]) / 2

    # try:
    #     # Plot time series at a specific point
    #     ts_fig = plot_time_series_at_point(ds, "speed", point_lat, point_lon)
    #
    #     # Save figure
    #     output_dir = create_output_dir()
    #     ts_output_path = os.path.join(
    #         output_dir, f"timeseries_{location_key}_{area_key}.png"
    #     )
    #     ts_fig.savefig(ts_output_path, dpi=300, bbox_inches="tight")
    #     print(f"Point time series saved to {ts_output_path}")
    # except Exception as e:
    #     print(f"Note: Could not create time series plot: {e}")

    return fig, ax


def visualize_distributions(histogram_data, output_dir=None, dpi=200):
    """
    Create and save distribution visualizations for variables across different locations.

    Parameters:
    -----------
    histogram_data : dict
        Dictionary with structure {variable_name: {'unit': str, 'data': {location_name: numpy_array_of_values}}}
    output_dir : str or Path, optional
        Directory to save visualizations. Default is "../analysis/viz/distributions"
    dpi : int, optional
        Resolution for saved figures. Default is 200.

    Returns:
    --------
    None
    """
    # Create visualizations directory if it doesn't exist
    if output_dir is None:
        output_dir = Path("../analysis/viz/distributions")
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Set up a consistent color palette using seaborn's "deep" palette
    all_locations = set()
    for var in histogram_data:
        for loc in histogram_data[var]["data"]:
            all_locations.add(loc)
    locations = list(all_locations)
    location_colors = dict(zip(locations, sns.color_palette("deep", len(locations))))

    # Set the font to Public Sans for all plots
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [
        "Public Sans",
        "DejaVu Sans",
        "Arial",
        "Helvetica",
    ]

    # For each variable, create visualization plots
    for variable, var_info in histogram_data.items():
        print(f"Creating distribution plots for {variable}...")

        # Get the unit for this variable
        unit = var_info.get("unit", "[m/s]")  # Default to [m/s] if not specified

        # Get the location data
        location_data = var_info["data"]

        # Skip if data for the variable is empty
        if not location_data:
            print(f"Skipping {variable} - no data available")
            continue

        # 1. COMBINED HISTOGRAMS (WITHOUT KDE)
        plt.figure(figsize=(14, 8))

        # Check if we have any data to plot
        has_data = False
        for location, values in location_data.items():
            if values is None or len(values) == 0:
                continue

            flat_values = values.flatten()
            flat_values = flat_values[~np.isnan(flat_values)]  # Remove NaN values

            if len(flat_values) == 0:
                continue

            has_data = True

            # Plot histogram WITHOUT KDE
            sns.histplot(
                flat_values,
                kde=False,  # No KDE line
                bins=50,
                color=location_colors[location],
                alpha=0.5,
                label=location,
                stat="density",
            )

        # Only add legend if we have data
        if has_data:
            plt.title(f"Histogram of {variable} by Location")
            plt.xlabel(f"{variable} {unit}")
            plt.ylabel("Density")

            # Only add legend if we have multiple locations
            if len(location_data) > 1:
                plt.legend()

            # Save combined histogram plot
            combined_path = output_dir / f"{variable}_combined_histogram.png"
            plt.savefig(combined_path, dpi=dpi)
        plt.close()

        # 2. COMBINED KDE PLOTS (WITHOUT HISTOGRAM)
        plt.figure(figsize=(14, 8))

        # Check if we have any data to plot
        has_data = False
        for location, values in location_data.items():
            if values is None or len(values) == 0:
                continue

            flat_values = values.flatten()
            flat_values = flat_values[~np.isnan(flat_values)]  # Remove NaN values

            if len(flat_values) == 0:
                continue

            has_data = True

            # Plot KDE only
            sns.kdeplot(
                flat_values,
                color=location_colors[location],
                label=location,
                linewidth=2,
            )

        # Only add legend if we have data
        if has_data:
            plt.title(f"Density Distribution of {variable} by Location")
            plt.xlabel(f"{variable} {unit}")
            plt.ylabel("Density")

            # Only add legend if we have multiple locations
            if len(location_data) > 1:
                plt.legend()

            # Save combined KDE plot
            combined_path = output_dir / f"{variable}_combined_kde.png"
            plt.savefig(combined_path, dpi=dpi)
        plt.close()

        # 3. INDIVIDUAL HISTOGRAM PLOTS (WITHOUT KDE)
        for location, values in location_data.items():
            if values is None or len(values) == 0:
                continue

            flat_values = values.flatten()
            flat_values = flat_values[~np.isnan(flat_values)]  # Remove NaN values

            if len(flat_values) == 0:
                continue

            plt.figure(figsize=(14, 8))

            # Plot histogram without KDE
            sns.histplot(
                flat_values,
                kde=False,
                bins=50,
                color=location_colors[location],
                alpha=0.7,
                stat="density",
            )

            plt.title(f"Histogram of {variable} in {location}")
            plt.xlabel(f"{variable} {unit}")
            plt.ylabel("Density")

            # Save individual histogram plot
            location_safe = location.replace(" ", "_").replace("/", "_")
            individual_path = output_dir / f"{variable}_{location_safe}_histogram.png"
            plt.savefig(individual_path, dpi=dpi)
            plt.close()

        # 4. INDIVIDUAL KDE PLOTS
        for location, values in location_data.items():
            if values is None or len(values) == 0:
                continue

            flat_values = values.flatten()
            flat_values = flat_values[~np.isnan(flat_values)]  # Remove NaN values

            if len(flat_values) == 0:
                continue

            plt.figure(figsize=(14, 8))

            # Plot KDE only
            sns.kdeplot(flat_values, color=location_colors[location], linewidth=2)

            plt.title(f"Density Distribution of {variable} in {location}")
            plt.xlabel(f"{variable} {unit}")
            plt.ylabel("Density")

            # Save individual KDE plot
            location_safe = location.replace(" ", "_").replace("/", "_")
            individual_path = output_dir / f"{variable}_{location_safe}_kde.png"
            plt.savefig(individual_path, dpi=dpi)
            plt.close()

        # 5. BOXPLOTS
        # Prepare data for boxplot
        boxplot_data = []
        boxplot_labels = []

        for location, values in location_data.items():
            if values is None or len(values) == 0:
                continue

            flat_values = values.flatten()
            flat_values = flat_values[~np.isnan(flat_values)]  # Remove NaN values

            if len(flat_values) == 0:
                continue

            # No sampling - use all data points
            boxplot_data.append(flat_values)
            boxplot_labels.append(location)

        # Only create boxplot if we have data
        if boxplot_data and boxplot_labels and len(boxplot_data) == len(boxplot_labels):
            fig = plt.figure(figsize=(14, 10))  # Increased height

            # Create boxplot with matching colors
            boxprops = dict(linewidth=2)
            flierprops = dict(marker="o", markerfacecolor="black", markersize=3)
            medianprops = dict(linewidth=2, color="black")

            # Create boxplot with fixed spacing
            plt.boxplot(
                boxplot_data,
                tick_labels=boxplot_labels,
                patch_artist=True,  # Fill boxes with color
                boxprops=boxprops,
                flierprops=flierprops,
                medianprops=medianprops,
            )

            # Color the boxes according to the location color palette
            for i, (patch, location) in enumerate(
                zip(plt.gca().patches, boxplot_labels)
            ):
                if i < len(boxplot_labels):  # Safety check
                    patch.set_facecolor(location_colors[location])

            plt.title(f"Boxplot of {variable} by Location")
            plt.ylabel(f"{variable} {unit}")
            plt.xticks(rotation=45, ha="right")

            # Explicitly set margins - avoid using tight_layout
            plt.subplots_adjust(bottom=0.3, top=0.9)

            # Save boxplot without tight_layout
            boxplot_path = output_dir / f"{variable}_boxplot.png"
            plt.savefig(boxplot_path, dpi=dpi)
            plt.close()
        else:
            print(
                f"Skipping boxplot for {variable} - dimensions of labels and data don't match"
            )

    # 6. SUBPLOT GRID BY LOCATION (HISTOGRAMS)
    # Create a subplot grid for each location showing all variables
    for location in locations:
        # Count how many variables are available for this location
        location_variables = []
        for var, var_info in histogram_data.items():
            if location in var_info["data"] and var_info["data"][location] is not None:
                flat_values = var_info["data"][location].flatten()
                flat_values = flat_values[~np.isnan(flat_values)]
                if len(flat_values) > 0:
                    location_variables.append(var)

        if not location_variables:
            print(f"Skipping location grid for {location} - no valid data")
            continue

        # Calculate grid dimensions - aim for roughly square layout
        n_plots = len(location_variables)
        n_cols = int(np.ceil(np.sqrt(n_plots)))
        n_rows = int(np.ceil(n_plots / n_cols))

        # Create figure and subplots for histograms
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 10))
        fig.suptitle(f"Histograms of Variables for {location}", fontsize=16)

        # Flatten axes array for easy indexing (even if it's 1D)
        if n_rows > 1 or n_cols > 1:
            axes = axes.flatten()
        else:
            axes = np.array([axes])

        # Plot each variable as histogram
        for i, variable in enumerate(location_variables):
            if i < len(axes):  # Safety check
                ax = axes[i]
                values = histogram_data[variable]["data"][location]
                unit = histogram_data[variable].get("unit", "[m/s]")
                flat_values = values.flatten()
                flat_values = flat_values[~np.isnan(flat_values)]  # Remove NaN values

                # Plot histogram without KDE
                sns.histplot(
                    flat_values,
                    kde=False,
                    bins=50,
                    color=location_colors[location],
                    alpha=0.7,
                    stat="density",
                    ax=ax,
                )

                ax.set_title(f"{variable} {unit}")
                ax.set_xlabel("")  # Remove individual xlabels

        # Hide any unused subplots
        for j in range(min(i + 1, len(axes))):
            if j >= len(location_variables):
                axes[j].set_visible(False)

        # Adjust layout without tight_layout
        plt.subplots_adjust(hspace=0.4, wspace=0.3, top=0.9, bottom=0.1)

        # Save combined location histograms
        location_safe = location.replace(" ", "_").replace("/", "_")
        location_path = output_dir / f"{location_safe}_all_histograms.png"
        plt.savefig(location_path, dpi=dpi)
        plt.close()

        # Create figure and subplots for KDEs
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 10))
        fig.suptitle(f"Density Distributions of Variables for {location}", fontsize=16)

        # Flatten axes array for easy indexing
        if n_rows > 1 or n_cols > 1:
            axes = axes.flatten()
        else:
            axes = np.array([axes])

        # Plot each variable as KDE
        for i, variable in enumerate(location_variables):
            if i < len(axes):  # Safety check
                ax = axes[i]
                values = histogram_data[variable]["data"][location]
                unit = histogram_data[variable].get("unit", "[m/s]")
                flat_values = values.flatten()
                flat_values = flat_values[~np.isnan(flat_values)]  # Remove NaN values

                # Plot KDE only
                sns.kdeplot(
                    flat_values, color=location_colors[location], linewidth=2, ax=ax
                )

                ax.set_title(f"{variable} {unit}")
                ax.set_xlabel("")  # Remove individual xlabels

        # Hide any unused subplots
        for j in range(min(i + 1, len(axes))):
            if j >= len(location_variables):
                axes[j].set_visible(False)

        # Adjust layout without tight_layout
        plt.subplots_adjust(hspace=0.4, wspace=0.3, top=0.9, bottom=0.1)

        # Save combined location KDEs
        location_safe = location.replace(" ", "_").replace("/", "_")
        location_path = output_dir / f"{location_safe}_all_kdes.png"
        plt.savefig(location_path, dpi=dpi)
        plt.close()

    # 7. VARIABLE COMPARISON MATRIX (BOTH HISTOGRAM AND KDE)
    # For each variable, create a subplot showing all locations
    for variable, var_info in histogram_data.items():
        # Get the unit
        unit = var_info.get("unit", "[m/s]")

        # Get location data
        location_data = var_info["data"]

        # Count locations for this variable with valid data
        var_locations = []
        for loc in location_data:
            if location_data[loc] is not None:
                flat_values = location_data[loc].flatten()
                flat_values = flat_values[~np.isnan(flat_values)]
                if len(flat_values) > 0:
                    var_locations.append(loc)

        if not var_locations:
            print(f"Skipping comparison matrix for {variable} - no valid data")
            continue

        # Calculate grid dimensions
        n_plots = len(var_locations)
        n_cols = int(np.ceil(np.sqrt(n_plots)))
        n_rows = int(np.ceil(n_plots / n_cols))

        # Create figure and subplots for histograms
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(14, 10), sharex=True, sharey=True
        )
        fig.suptitle(
            f"Histogram Comparison of {variable} Across Locations", fontsize=16
        )

        # Flatten axes array for easy indexing
        if n_rows > 1 or n_cols > 1:
            axes = axes.flatten()
        else:
            axes = np.array([axes])

        # Plot each location as histogram
        for i, loc in enumerate(var_locations):
            if i < len(axes):  # Safety check
                ax = axes[i]
                values = location_data[loc]
                flat_values = values.flatten()
                flat_values = flat_values[~np.isnan(flat_values)]  # Remove NaN values

                # Plot histogram without KDE
                sns.histplot(
                    flat_values,
                    kde=False,
                    bins=50,
                    color=location_colors[loc],
                    alpha=0.7,
                    stat="density",
                    ax=ax,
                )

                ax.set_title(loc)

        # Hide any unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        # Add a common x-label with unit
        fig.text(0.5, 0.04, f"{variable} {unit}", ha="center", fontsize=12)

        # Add a common y-label
        fig.text(0.04, 0.5, "Density", va="center", rotation="vertical", fontsize=12)

        # Adjust layout
        plt.subplots_adjust(
            hspace=0.4, wspace=0.3, top=0.9, bottom=0.15, left=0.1, right=0.95
        )

        # Save histogram comparison plot
        comparison_path = output_dir / f"{variable}_location_histogram_comparison.png"
        plt.savefig(comparison_path, dpi=dpi)
        plt.close()

        # Create figure and subplots for KDEs
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(14, 10), sharex=True, sharey=True
        )
        fig.suptitle(
            f"Density Distribution Comparison of {variable} Across Locations",
            fontsize=16,
        )

        # Flatten axes array for easy indexing
        if n_rows > 1 or n_cols > 1:
            axes = axes.flatten()
        else:
            axes = np.array([axes])

        # Plot each location as KDE
        for i, loc in enumerate(var_locations):
            if i < len(axes):  # Safety check
                ax = axes[i]
                values = location_data[loc]
                flat_values = values.flatten()
                flat_values = flat_values[~np.isnan(flat_values)]  # Remove NaN values

                # Plot KDE only
                sns.kdeplot(flat_values, color=location_colors[loc], linewidth=2, ax=ax)

                ax.set_title(loc)

        # Hide any unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        # Add a common x-label with unit
        fig.text(0.5, 0.04, f"{variable} {unit}", ha="center", fontsize=12)

        # Add a common y-label
        fig.text(0.04, 0.5, "Density", va="center", rotation="vertical", fontsize=12)

        # Adjust layout
        plt.subplots_adjust(
            hspace=0.4, wspace=0.3, top=0.9, bottom=0.15, left=0.1, right=0.95
        )

        # Save KDE comparison plot
        comparison_path = output_dir / f"{variable}_location_kde_comparison.png"
        plt.savefig(comparison_path, dpi=dpi)
        plt.close()

    print("All distribution visualizations complete!")


def main():
    # Create the visualizer
    visualizer = TidalVisualizer(viz_config)

    PLOT_MAPS = False

    config = {
        "AK_cook_inlet": {
            "label": "Cook Inlet, Alaska",
            # "path": "../data/AK_aleutian_islands/b2_summary/001.AK_aleutian_islands.tidal_hindcast_fvcom-1_year_average.b2.20100603.000000.nc",
            "path": "/scratch/asimms/Tidal/AK_cook_inlet/b2_summary_vap/001.AK_cook_inlet.tidal_hindcast_fvcom-1_year_average.b2.20050101.000000.nc",
            "zoom": 10,
        },
        "AK_aleutian_islands": {
            "label": "Aleutian Islands, Alaska",
            # "path": "../data/AK_aleutian_islands/b2_summary/001.AK_aleutian_islands.tidal_hindcast_fvcom-1_year_average.b2.20100603.000000.nc",
            "path": "/scratch/asimms/Tidal/AK_aleutian_islands/b2_summary_vap/001.AK_aleutian_islands.tidal_hindcast_fvcom-1_year_average.b2.20100603.000000.nc",
            "zoom": 2,
        },
        "ME_western_passage": {
            "label": "Western Passage, Maine",
            # "path": "../data/ME_western_passage/b2_summary/001.ME_western_passage.tidal_hindcast_fvcom-1_year_average.b2.20170101.000000.nc",
            "path": "/scratch/asimms/Tidal/ME_western_passage/b2_summary_vap/001.ME_western_passage.tidal_hindcast_fvcom-1_year_average.b2.20170101.000000.nc",
            "zoom": 10,
        },
        "NH_piscataqua_river": {
            "label": "Piscataqua River, New Hampshire",
            # "path": "../data/NH_piscataqua_river/b2_summary/001.NH_piscataqua_river.tidal_hindcast_fvcom-1_year_average.b2.20070101.000000.nc",
            "path": "/scratch/asimms/Tidal/NH_piscataqua_river/b2_summary_vap/001.NH_piscataqua_river.tidal_hindcast_fvcom-1_year_average.b2.20070101.000000.nc",
            "zoom": 13,
        },
        "WA_puget_sound": {
            "label": "Puget Sound, Washington",
            # "path": "../data/WA_puget_sound/b2_summary/001.WA_puget_sound.tidal_hindcast_fvcom-1_year_average.b2.20150101.000000.nc",
            "path": "/scratch/asimms/Tidal/WA_puget_sound/b2_summary_vap/001.WA_puget_sound.tidal_hindcast_fvcom-1_year_average.b2.20150101.000000.nc",
            "zoom": 10,
        },
    }

    variables = [
        # "u_depth_avg",
        # "v_depth_avg",
        # "to_direction_depth_avg",
        # "from_direction_depth_avg",
        # "speed_depth_avg",
        "vap_water_column_mean_sea_water_speed",
        "vap_water_column_mean_sea_water_power_density",
        "vap_water_column_mean_element_volume_energy_flux",
        # "speeddepth_median",
        # "speed_depth_95th_percentile",
        # "speed_depth_max",
        # "power_density_depth_avg",
        # "power_densitydepth_median",
        # "power_density_depth_95th_percentile",
        # "power_density_depth_max",
        # "volume_energy_flux_depth_avg",
        # "volume_energy_fluxdepth_median",
        # "volume_energy_flux_depth_95th_percentile",
        # "volume_energy_flux_depth_max",
        # "vertical_avg_energy_flux",
        # "seafloor_depth",
        # "element_volume",
    ]

    old_variables = [
        # "u_depth_avg",
        # "v_depth_avg",
        # "to_direction_depth_avg",
        # "from_direction_depth_avg",
        "speed_depth_avg",
        # "vap_water_column_mean_sea_water_speed",
        # "vap_water_column_mean_sea_water_power_density",
        # "vap_water_column_mean_element_volume_energy_flux",
        # "speeddepth_median",
        # "speed_depth_95th_percentile",
        # "speed_depth_max",
        "power_density_depth_avg",
        # "power_densitydepth_median",
        # "power_density_depth_95th_percentile",
        # "power_density_depth_max",
        "volume_energy_flux_depth_avg",
        # "volume_energy_fluxdepth_median",
        # "volume_energy_flux_depth_95th_percentile",
        # "volume_energy_flux_depth_max",
        # "vertical_avg_energy_flux",
        # "seafloor_depth",
        # "element_volume",
    ]

    # Used to correct the old puget sound variable names for the distribution visualisation
    variable_name_map = {
        "speed_depth_avg": {
            "label": "Sea Water Speed",
            "unit": "[m/s]",
        },
        "power_density_depth_avg": {
            "label": "Sea Water Power Density",
            "unit": "[W/m^2]",
        },
        "volume_energy_flux_depth_avg": {
            "label": "Sea Water Column Energy Flux",
            "unit": "[W]",
        },
        "vap_water_column_mean_sea_water_speed": {
            "label": "Sea Water Speed",
            "unit": "[m/s]",
        },
        "vap_water_column_mean_sea_water_power_density": {
            "label": "Sea Water Power Density",
            "unit": "[W/m^2]",
        },
        "vap_water_column_mean_element_volume_energy_flux": {
            "label": "Sea Water Column Energy Flux",
            "unit": "[W]",
        },
    }

    histogram_data = {}

    for key, location in config.items():
        ds = xr.open_dataset(location["path"])
        # for variable_name in sorted(list(ds.variables.keys())):
        # print(variable_name)
        output_dir = Path(f"../analysis/viz/{key}").resolve()
        # output_dir = Path(f"/scratch/asimms/Tidal/viz/{key}/complete_site")

        output_dir.mkdir(exist_ok=True, parents=True)

        if key == "WA_puget_sound":
            this_variables = old_variables
        else:
            this_variables = variables

        for variable in this_variables:
            print(f"Visualizing {key} {variable}...")

            cbar_min = None
            cbar_max = None

            speed_max = 1.75  # m/s
            if "speed" in variable:
                cbar_min = 0
                cbar_max = speed_max

            if "power_density" in variable:
                cbar_min = 0
                # ds[output_variable_name] = 0.5 * rho * ds[output_names["speed"]] ** 3
                rho = 1025
                power_density_max = 0.5 * rho * (speed_max**3)
                # Round to the nearest thousand
                cbar_max = round(power_density_max, -3)

            if PLOT_MAPS is True:
                fig, ax = visualizer.create_tidal_visualization(
                    ds,
                    variable,
                    display_name=location["label"],
                    zoom_level=location["zoom"],
                    cbar_min=cbar_min,
                    cbar_max=cbar_max,
                )

                # Save figure
                output_path = Path(
                    output_dir, f"wpto_hr_tidal_hindcast_{key}_{variable}_v006.png"
                )
                small_output_path = Path(
                    output_dir, f"wpto_hr_tidal_hindcast_{key}_{variable}_web_v006.png"
                )

                print(f"Saving {key} {variable} to {output_path}...")
                plt.savefig(output_path, dpi=300, bbox_inches="tight")
                # print(f"Saving {key} {variable} to {small_output_path}...")
                plt.savefig(small_output_path, dpi=80, bbox_inches="tight")
                print("Save successful!")
                plt.close(fig)

            var_label = variable_name_map[variable]["label"]
            var_unit = variable_name_map[variable]["unit"]
            location_label = location["label"]

            # if variable not in histogram_data:
            if var_label not in histogram_data:
                histogram_data[var_label] = {
                    "unit": var_unit,
                    "data": {},
                }

            if location_label not in histogram_data[var_label]["data"]:
                histogram_data[var_label]["data"][location_label] = ds[variable].values

        ds = ds.close()

    visualize_distributions(histogram_data)

    # Plot combined distribution and histograms for each variable

    # Your code here, use seaborn

    # print("Available locations:")
    # locations = get_location_names(viz_config)
    # for location in locations:
    #     print(f"  - {location}")
    #     areas = get_areas_of_interest(viz_config, location)
    #     for area in areas:
    #         print(f"      - {area}")

    # Select a location and area for examples
    # location_key = "AK_cook_inlet"
    # location_key = "AK_aleutian_islands"
    # location_key = "WA_puget_sound"
    # variable = "seafloor_depth"
    # area_key = "western_passage_main"
    # area_key = "all"
    # area_key = None

    # Example 1: Basic visualization (using the original functionality)
    # print("\nCreating basic visualization...")
    # fig, ax = visualizer.visualize_area_of_interest(
    #     location_key=location_key, area_key=area_key, variable_name="speed"
    # )
    # ds = xr.open_dataset(
    #     # "../data/AK_aleutian_islands/b2_summary/001.AK_aleutian_islands.tidal_hindcast_fvcom-1_year_average.b2.20100603.000000.nc"
    #     "../data/WA_puget_sound/b2_summary/001.WA_puget_sound.tidal_hindcast_fvcom-1_year_average.b2.20150101.000000.nc"
    # )
    #
    # print(f"Visualizing {location_key} {area_key}...")
    #
    # fig, ax = visualizer.create_tidal_visualization(
    #     ds, variable, display_name="Puget Sound"
    # )
    #
    # # Save figure
    # output_dir = create_output_dir()
    # output_path = os.path.join(
    #     output_dir, f"basic_{location_key}_{area_key}_{variable}.png"
    # )
    # print(f"Saving {location_key} {area_key} {variable}...")
    # plt.savefig(output_path, dpi=300, bbox_inches="tight")
    # print(f"Basic visualization saved to {output_path}")
    # plt.close(fig)

    # Example 2: Enhanced area visualization
    # print("Creating enhanced area visualization")
    # fig, ax = example_enhanced_area_visualization(visualizer, location_key, area_key)
    #
    # output_path = os.path.join(output_dir, f"enhanced_{location_key}_{area_key}.png")
    # plt.savefig(output_path, dpi=300, bbox_inches="tight")
    # print(f"Enhanced visualization saved to {output_path}")
    # plt.close(fig)
    #
    # # Example 3: Time series visualization
    # try:
    #     fig, ax = example_time_series(visualizer, location_key, area_key)
    #     plt.close(fig)
    # except Exception as e:
    #     print(f"Could not create time series visualization: {e}")
    #
    # print("\nAll examples completed!")


if __name__ == "__main__":
    main()
