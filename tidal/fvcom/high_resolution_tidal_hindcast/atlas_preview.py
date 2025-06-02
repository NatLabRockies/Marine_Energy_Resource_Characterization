from pathlib import Path
from typing import Any, Dict, List, Union

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmocean
import contextily as ctx
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import pandas as pd
import seaborn as sns

from pyproj import Transformer

# Set the base directory - modify this to match your system
BASE_DIR = Path("/projects/hindcastra/Tidal/datasets/high_resolution_tidal_hindcast")

VIZ_OUTPUT_DIR = Path("/home/asimms/tidal/analysis/viz/")
VIZ_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEA_WATER_SPEED_CBAR_MAX = 2.0
SEA_WATER_SPEED_CBAR_MIN = 0.0
# In the output visualization, there will be 9 levels, but this is the number of levels within the range
# The 9th level is for values outside the range.
SEA_WATER_SPEED_LEVELS = 8

SEA_WATER_MAX_SPEED_CBAR_MAX = 5.0
SEA_WATER_MAX_SPEED_LEVELS = 10

SEA_WATER_POWER_DENSITY_CBAR_MAX = 4000  # 0.5 * 1025 * (2.0^3) = 4,100
SEA_WATER_POWER_DENSITY_CBAR_MIN = 0
SEA_WATER_POWER_DENSITY_LEVELS = 8

SEA_WATER_MAX_POWER_DENSITY_CBAR_MAX = 64000  # 0.5 * 1025 * (5.0^3) = 64,062.5
SEA_WATER_MAX_POWER_DENSITY_LEVELS = 8

SEA_FLOOR_DEPTH_MIN = 0
SEA_FLOOR_DEPTH_MAX = 200
SEA_FLOOR_DEPTH_LEVELS = 10

BASEMAP_PROVIDER = ctx.providers.Esri.WorldImagery

SEA_WATER_SPEED_UNITS = r"$m/s$"
SEA_WATER_POWER_DENSITY_UNITS = r"$W/m^2$"

# Note the output visualization will actually have 9 levels
# There will be 8 within the range and a 9th that is outside of the range
# This makes the range easier for the user understand and interpret
COLOR_BAR_DISCRETE_LEVELS = 8

MEAN_SPEED_CMAP = cmocean.cm.thermal
MAX_SPEED_CMAP = cmocean.cm.matter
MEAN_POWER_DENSITY_CMAP = cmocean.cm.dense
# MAX_POWER_DENSITY_CMAP = cmocean.cm.tempo
MAX_POWER_DENSITY_CMAP = cmocean.cm.amp
SEA_FLOOR_DEPTH_CMAP = cmocean.cm.deep


# Define available regions (derived from folder structure)
def get_available_regions():
    """Get list of available regions based on directory structure"""
    return sorted([d.name for d in BASE_DIR.iterdir() if d.is_dir()])


# Function to get parquet file path for a given region
def get_parquet_path(region):
    """Get the path to the parquet file for the specified region"""
    parquet_dir = BASE_DIR / region / "b6_vap_atlas_summary_parquet"
    parquet_files = sorted(list(parquet_dir.glob("*.parquet")))

    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found for region: {region}")

    # Return the first parquet file (there should be only one per region based on the structure)
    return parquet_files[0]


def plot_tidal_variable(
    df,
    location,
    column_name,
    label,
    units,
    vmin,
    vmax,
    cmap="viridis",
    figsize=(24, 18),
    point_size=1,
    alpha=0.99,
    title=None,
    is_aleutian=False,
    plot_type="mesh",
    line_width=0,
    show=False,
    save_path=None,
    n_colors=COLOR_BAR_DISCRETE_LEVELS,
):
    """
    Plot tidal variables with optional discrete color levels.

    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing the tidal data
    location : str
        Location identifier
    column_name : str
        Name of the column in df to plot
    label : str
        Label for the colorbar
    units : str
        Units for the colorbar
    vmin, vmax : float
        Minimum and maximum values for the colorbar
    cmap : str, default="viridis"
        Colormap to use
    figsize : tuple, default=(16, 12)
        Figure size
    point_size : int, default=1
        Size of points for point plots
    alpha : float, default=0.9
        Transparency of the plot
    title : str, default=None
        Title for the plot
    is_aleutian : bool, default=False
        Whether to use Aleutian projection
    plot_type : str, default="mesh"
        Type of plot, either "mesh" or "points"
    line_width : int, default=0
        Width of lines for mesh plots
    n_colors : int, default=None
        Number of discrete color levels. If None, continuous colormap is used.
    """

    fig = plt.figure(figsize=figsize)

    has_mesh_data = _check_mesh_data_availability(df)

    if plot_type == "mesh" and not has_mesh_data:
        print(
            "Warning: Requested mesh plot but element corner data not found. Falling back to point plot."
        )
        plot_type = "points"

    lon_min, lon_max, lat_min, lat_max, lon_padding, lat_padding = (
        _calculate_coordinate_bounds(df)
    )

    # Create a discrete colormap if n_colors is specified
    discrete_norm = None
    if n_colors is not None:
        # Create discrete colormap with n_colors + 1 levels (including the above-max level)
        base_cmap = plt.get_cmap(cmap)

        # Get n_colors + 1 colors evenly spaced from the colormap
        colors = base_cmap(np.linspace(0, 1, n_colors + 1))

        # Create a new colormap with n_colors + 1 levels
        discrete_cmap = mpl.colors.LinearSegmentedColormap.from_list(
            "discrete_cmap", colors, N=n_colors + 1
        )

        # Create boundaries for n_colors discrete levels within vmin-vmax
        main_bounds = np.linspace(vmin, vmax, n_colors + 1)

        # Add an extra boundary for values above vmax
        very_large_value = vmax * 1000  # Much larger than vmax but still finite
        bounds = np.append(main_bounds, very_large_value)

        # Create a BoundaryNorm with n_colors + 1 levels
        discrete_norm = mpl.colors.BoundaryNorm(bounds, n_colors + 1)
        cmap = discrete_cmap
    else:
        bounds = None
        main_bounds = None

    if is_aleutian:
        ax, projection = _setup_aleutian_projection(
            df, lon_min, lon_max, lat_min, lat_max
        )
        transform = ccrs.PlateCarree()
    else:
        ax, x, y, transformer = _setup_standard_projection(
            df, location, lon_min, lon_max, lat_min, lat_max
        )
        transform = None

    if is_aleutian:
        if plot_type == "mesh":
            scatter = _plot_aleutian_mesh_with_triangulation(
                ax,
                df,
                column_name,
                cmap,
                alpha,
                vmin,
                vmax,
                line_width,
                transform,
                point_size,
                norm=discrete_norm,
            )
        else:
            scatter = _plot_aleutian_points(
                ax,
                df,
                column_name,
                cmap,
                point_size,
                alpha,
                vmin,
                vmax,
                transform,
                norm=discrete_norm,
            )
    else:
        if plot_type == "mesh":
            scatter = _plot_standard_mesh_with_triangulation(
                ax,
                df,
                column_name,
                cmap,
                alpha,
                vmin,
                vmax,
                line_width,
                x,
                y,
                transformer,
                norm=discrete_norm,
            )
        else:
            scatter = _plot_standard_points(
                ax,
                df,
                column_name,
                cmap,
                point_size,
                alpha,
                vmin,
                vmax,
                x,
                y,
                norm=discrete_norm,
            )

    # Pass only the main bounds to _add_colorbar_and_title
    _add_colorbar_and_title(
        fig,
        ax,
        scatter,
        label,
        units,
        title,
        location,
        discrete_levels=main_bounds if n_colors is not None else None,
    )

    plt.tight_layout()
    if show is True:
        plt.show()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    # Print out the color and value range for each level if discrete levels are used
    color_data = None
    if n_colors is not None:
        # Only print the main bounds and the above-max level
        color_data = _print_color_level_ranges(
            main_bounds, label, units, discrete_cmap, n_colors + 1
        )

    return fig, ax, color_data


def _check_mesh_data_availability(df):
    required_columns = [
        "element_corner_1_lat",
        "element_corner_1_lon",
        "element_corner_2_lat",
        "element_corner_2_lon",
        "element_corner_3_lat",
        "element_corner_3_lon",
    ]
    return all(col in df.columns for col in required_columns)


def _calculate_coordinate_bounds(df):
    lon_min, lon_max = df["lon_center"].min(), df["lon_center"].max()
    lat_min, lat_max = df["lat_center"].min(), df["lat_center"].max()

    lon_padding = max(0.5, (lon_max - lon_min) * 0.1)
    lat_padding = max(0.5, (lat_max - lat_min) * 0.1)

    return lon_min, lon_max, lat_min, lat_max, lon_padding, lat_padding


def _setup_aleutian_projection(df, lon_min, lon_max, lat_min, lat_max):
    if (lon_max - lon_min) > 180:
        central_lon = -175
        central_lat = 52
    else:
        central_lon = (lon_min + lon_max) / 2
        central_lat = (lat_min + lat_max) / 2

    projection = ccrs.Orthographic(
        central_longitude=central_lon, central_latitude=central_lat
    )

    ax = plt.axes(projection=projection)

    _add_aleutian_map_features(ax)

    return ax, projection


def _add_aleutian_map_features(ax):
    land = cfeature.NaturalEarthFeature(
        "physical",
        "land",
        "10m",
        edgecolor="black",
        facecolor="lightgray",
        linewidth=0.25,
    )
    ax.add_feature(land)

    ocean = cfeature.NaturalEarthFeature(
        "physical", "ocean", "10m", edgecolor="none", facecolor="lightblue", alpha=0.4
    )
    ax.add_feature(ocean)

    coastline = cfeature.NaturalEarthFeature(
        "physical",
        "coastline",
        "10m",
        edgecolor="black",
        facecolor="none",
        linewidth=0.6,
    )
    ax.add_feature(coastline)

    borders = cfeature.NaturalEarthFeature(
        "cultural",
        "admin_0_boundary_lines_land",
        "10m",
        edgecolor="gray",
        facecolor="none",
        linewidth=0.3,
    )
    ax.add_feature(borders)

    lakes = cfeature.NaturalEarthFeature(
        "physical",
        "lakes",
        "10m",
        edgecolor="black",
        facecolor="lightblue",
        linewidth=0.25,
    )
    ax.add_feature(lakes)

    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=0.5,
        color="gray",
        alpha=0.5,
        linestyle="--",
    )
    gl.top_labels = False
    gl.right_labels = False


def _setup_standard_projection(df, location, lon_min, lon_max, lat_min, lat_max):
    zoom_levels = {
        "AK_cook_inlet": 9,
        "ME_western_passage": 10,
        "NH_piscataqua_river": 11,
        "WA_puget_sound": 10,
    }

    zoom_level = zoom_levels.get(location, 9)
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    x, y = transformer.transform(df["lon_center"].values, df["lat_center"].values)

    ax = plt.subplot(111)

    _setup_standard_plot_appearance(ax, x, y, zoom_level)

    return ax, x, y, transformer


def _setup_standard_plot_appearance(ax, x, y, zoom_level):
    ax.set_aspect("equal")

    x_buffer = (max(x) - min(x)) * 0.15
    y_buffer = (max(y) - min(y)) * 0.15

    ax.set_xlim(min(x) - x_buffer, max(x) + x_buffer)
    ax.set_ylim(min(y) - y_buffer, max(y) + y_buffer)

    ax.set_axis_off()

    ax.grid(False)

    ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery, zoom=zoom_level)


def _create_triangulation_from_element_corners(df, transformer):
    corners_lon = np.column_stack(
        [
            df["element_corner_1_lon"].values,
            df["element_corner_2_lon"].values,
            df["element_corner_3_lon"].values,
        ]
    ).flatten()

    corners_lat = np.column_stack(
        [
            df["element_corner_1_lat"].values,
            df["element_corner_2_lat"].values,
            df["element_corner_3_lat"].values,
        ]
    ).flatten()

    corners_x, corners_y = transformer.transform(corners_lon, corners_lat)

    num_triangles = len(df)
    triangles = np.arange(num_triangles * 3).reshape(-1, 3)

    triang = mtri.Triangulation(corners_x, corners_y, triangles)

    return triang, corners_x, corners_y


def _plot_aleutian_points(
    ax, df, column_name, cmap, point_size, alpha, vmin, vmax, transform, norm=None
):
    scatter = ax.scatter(
        df["lon_center"],
        df["lat_center"],
        c=df[column_name],
        cmap=cmap,
        s=point_size,
        alpha=alpha,
        vmin=vmin,
        vmax=vmax,
        transform=transform,
        norm=norm,
    )
    return scatter


def _plot_aleutian_mesh_with_triangulation(
    ax,
    df,
    column_name,
    cmap,
    alpha,
    vmin,
    vmax,
    line_width,
    transform,
    point_size=5,
    norm=None,
):
    corners_lon = np.column_stack(
        [
            df["element_corner_1_lon"].values,
            df["element_corner_2_lon"].values,
            df["element_corner_3_lon"].values,
        ]
    ).flatten()

    corners_lat = np.column_stack(
        [
            df["element_corner_1_lat"].values,
            df["element_corner_2_lat"].values,
            df["element_corner_3_lat"].values,
        ]
    ).flatten()

    lon_range = np.max(corners_lon) - np.min(corners_lon)
    if lon_range > 180:
        negative_lons = np.sum(corners_lon < 0)
        positive_lons = np.sum(corners_lon > 0)

        if negative_lons > positive_lons:
            corners_lon = np.where(corners_lon > 0, corners_lon - 360, corners_lon)
        else:
            corners_lon = np.where(corners_lon < 0, corners_lon + 360, corners_lon)

    # Do NOT remove this
    # This is a rendering hint to plot this location correctly.
    # This places extent of the map in the correct location
    point_scatter = ax.scatter(
        df["lon_center"],
        df["lat_center"],
        c="red",
        s=10,
        alpha=0.0,
        transform=transform,
        edgecolor="none",
    )

    triangles = np.arange(len(df) * 3).reshape(-1, 3)

    triang = mtri.Triangulation(corners_lon, corners_lat, triangles)

    data_values = np.repeat(df[column_name].values, 3)

    try:
        tcf = ax.tripcolor(
            triang,
            data_values,
            cmap=cmap,
            alpha=alpha,
            # vmin=vmin,
            # vmax=vmax,
            transform=transform,
            norm=norm,
        )

        if line_width > 0:
            ax.triplot(
                triang,
                color="black",
                linewidth=line_width,
                alpha=0.5,
                transform=transform,
            )
    except Exception:
        tcf = ax.scatter(
            df["lon_center"],
            df["lat_center"],
            c=df[column_name],
            cmap=cmap,
            s=point_size,
            alpha=alpha,
            # vmin=vmin,
            # vmax=vmax,
            transform=transform,
            norm=norm,
        )

    return tcf


def _plot_standard_points(
    ax, df, column_name, cmap, point_size, alpha, vmin, vmax, x, y, norm=None
):
    scatter = ax.scatter(
        x,
        y,
        c=df[column_name],
        cmap=cmap,
        s=point_size,
        alpha=alpha,
        # vmin=vmin,
        # vmax=vmax,
        norm=norm,
    )
    return scatter


def _plot_standard_mesh_with_triangulation(
    ax,
    df,
    column_name,
    cmap,
    alpha,
    vmin,
    vmax,
    line_width,
    x,
    y,
    transformer,
    norm=None,
):
    triang, corners_x, corners_y = _create_triangulation_from_element_corners(
        df, transformer
    )

    mesh_data = np.repeat(df[column_name].values, 3)

    scatter = ax.tripcolor(
        triang,
        mesh_data,
        cmap=cmap,
        alpha=alpha,
        norm=norm,  # vmin=vmin, vmax=vmax, norm=norm
    )

    if line_width > 0:
        ax.triplot(triang, color="black", linewidth=line_width, alpha=0.3)

    return scatter


def _add_colorbar_and_title(
    fig, ax, scatter, label, units, title, location, discrete_levels=None
):
    """
    Add colorbar and title to the plot with optional discrete levels.
    Parameters:
    -----------
    fig : matplotlib Figure
        Figure object
    ax : matplotlib Axes
        Axes object
    scatter : matplotlib collection
        Collection object returned by scatter or tripcolor
    label : str
        Label for the colorbar
    units : str
        Units for the colorbar
    title : str or None
        Title for the plot
    location : str
        Location identifier
    discrete_levels : array-like, default=None
        Discrete level boundaries for the colorbar
    """
    # Create a colorbar with discrete ticks if discrete levels are provided
    if discrete_levels is not None:
        # Format the tick labels based on the number of decimals needed
        max_value = max(abs(discrete_levels.min()), abs(discrete_levels.max()))
        if max_value >= 1000:
            tick_format = "%.0f"
        elif max_value >= 100:
            tick_format = "%.1f"
        elif max_value >= 10:
            tick_format = "%.2f"
        else:
            tick_format = "%.3f"

        # Calculate the interval between levels
        interval = (discrete_levels[-1] - discrete_levels[0]) / (
            len(discrete_levels) - 1
        )

        # Create the colorbar
        cbar = fig.colorbar(
            scatter, ax=ax, orientation="vertical", pad=0.02, fraction=0.03, shrink=0.7
        )

        # Create ranges from the discrete levels
        ranges = []
        for i in range(len(discrete_levels) - 1):
            start = discrete_levels[i]
            end = discrete_levels[i + 1]
            ranges.append((start, end))

        # Calculate midpoints for tick positions
        midpoints = [(r[0] + r[1]) / 2 for r in ranges]

        # Create labels showing range intervals
        tick_labels = []
        for i, (start, end) in enumerate(ranges):
            tick_labels.append(f"[{tick_format % start}-{tick_format % end})")

        # Add position and label for the "above max" range
        # Position it one interval higher (at the same distance as other ticks)
        above_max_midpoint = discrete_levels[-1] + interval / 2
        midpoints.append(above_max_midpoint)

        # Add the final "≥ max_value" label
        tick_labels.append(f"[≥{tick_format % discrete_levels[-1]})")

        print(f"Midpoints: {midpoints}")
        print(f"Tick labels: {tick_labels}")

        # First extend the colorbar's axis limits to include our new tick
        # We need to do this BEFORE setting the ticks
        ymin, ymax = cbar.ax.get_ylim()
        new_ymax = max(ymax, above_max_midpoint + interval / 2)
        cbar.ax.set_ylim(ymin, new_ymax)

        # Now set the ticks and labels
        cbar.ax.yaxis.set_ticks(midpoints)
        cbar.ax.yaxis.set_ticklabels(tick_labels)

        # Print the limits after adjustment
        print(f"Colorbar y-limits after adjustment: {cbar.ax.get_ylim()}")

    else:
        # Standard continuous colorbar
        cbar = fig.colorbar(
            scatter, ax=ax, orientation="vertical", pad=0.02, fraction=0.03, shrink=0.7
        )

    cbar.set_label(f"{label} [{units}]")
    if title is None:
        title = f"{location.replace('_', ' ').title()} - {label}"
    plt.title(title)


# Define accurate column display names
COLUMN_DISPLAY_NAMES = {
    "vap_water_column_mean_sea_water_speed": "Water Column Mean Speed",
    "vap_water_column_95th_percentile_sea_water_speed": "Water Column 95th Percentile Speed",
    "vap_water_column_max_sea_water_speed": "Water Column Max Speed",
}


def analyze_variable(
    df: pd.DataFrame,
    variable: str,
    variable_display_name,
    region_name,
    units: str = "m/s",
    percentiles: List[float] = [0.95, 0.99, 0.9999],
    output_path: Union[Path, None] = None,
) -> Dict[str, Any]:
    """
    Analyze a variable's statistics and plot histogram with KDE and percentile lines.

    Args:
        df: DataFrame containing the variable data
        variable: Column name of the variable to analyze
        variable_display_name: Display name for the variable (defaults to variable name if None)
        region_name: Name of the region/area being analyzed (optional)
        units: Units for the variable (for axis labels)
        percentiles: List of percentiles to calculate and display
        output_path: Path to save the output plots (if None, plots are displayed)

    Returns:
        Dictionary containing statistics and metadata for meta-analysis
    """
    # Set display name if not provided
    if variable_display_name is None:
        variable_display_name = variable

    # Set region name if not provided
    region_label = f"{region_name} - " if region_name else ""

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 8))

    # Dictionary to store the results
    results = {
        "variable": variable,
        "variable_display_name": variable_display_name,
        "region": region_name,
        "units": units,
        "df": df,  # Store the dataframe for use in the meta-analysis
        "stats": {},
    }

    # Calculate key percentiles
    percentile_values = {}
    for p in percentiles:
        p_label = f"p{int(p*100)}" if p < 1 else f"p{int(p*10000)/100}"
        percentile_values[p_label] = df[variable].quantile(p)

    # Add min and max
    percentile_values["min"] = df[variable].min()
    percentile_values["max"] = df[variable].max()

    # Store statistics
    results["stats"] = percentile_values

    # Plot histogram with KDE
    sns.histplot(df[variable], kde=True, ax=ax, bins=50, alpha=0.6)

    # Add vertical lines for percentiles and annotations
    colors = ["r", "g", "b", "purple", "orange"]  # Add more if needed

    y_max = ax.get_ylim()[1]
    y_positions = np.linspace(0.9, 0.7, len(percentiles))

    for i, p in enumerate(percentiles):
        p_label = f"p{int(p*100)}" if p < 1 else f"p{int(p*10000)/100}"
        value = percentile_values[p_label]

        # Add vertical line
        ax.axvline(value, color=colors[i % len(colors)], linestyle="--")

        # Format percentile for display
        display_p = f"{p*100:.0f}%" if p < 1 else f"{p*100:.2f}%"

        # Add annotation
        ax.annotate(
            f"{display_p}: {value:.3f}",
            xy=(value, 0),
            xytext=(value, y_max * y_positions[i]),
            arrowprops=dict(arrowstyle="->"),
            ha="right" if i < len(percentiles) // 2 else "left",
        )

    # Set title and labels
    ax.set_title(f"{region_label}{variable_display_name} ({units})")
    ax.set_xlabel(f"{variable_display_name} ({units})")
    ax.set_ylabel("Frequency")

    # Tight layout and show/save
    plt.tight_layout()
    if output_path is not None:
        plt.savefig(Path(output_path, f"{variable}_analysis.png"))
    else:
        plt.show()

    return results


def analyze_variable_across_regions(
    region_stats: List[Dict[str, Any]], output_path: Union[Path, None] = None
) -> Dict[str, Any]:
    """
    Perform meta-analysis comparing variable statistics across different regions.

    Args:
        region_stats: List of dictionaries containing region statistics from analyze_variable
        output_path: Path to save the output plots (if None, plots are displayed)

    Returns:
        Dictionary with summary statistics and comparison tables
    """
    if not region_stats:
        print("No region statistics provided for analysis.")
        return {}

    # Extract variable info from the first region stats
    variable = region_stats[0]["variable"]
    variable_display_name = region_stats[0]["variable_display_name"]
    units = region_stats[0].get("units", "")

    # Extract regions
    regions = [stat["region"] for stat in region_stats if stat["region"] is not None]

    # Create consistent color mapping for regions
    cmap = plt.get_cmap("tab10", len(regions))
    region_colors = {region: cmap(i) for i, region in enumerate(regions)}

    # Create comparison table
    table_data = []
    metrics = set()

    for stat in region_stats:
        if stat["region"] is not None:  # Skip if region is None
            row = {"Region": stat["region"]}

            # Add all metrics from this region
            for metric, value in stat["stats"].items():
                row[metric] = value
                metrics.add(metric)

            table_data.append(row)

    # Create DataFrame
    comparison_table = pd.DataFrame(table_data)

    # Plot KDE comparison
    plt.figure(figsize=(16, 8))

    # Plot KDE for each region
    for stat in region_stats:
        if stat["region"] is not None and "df" in stat:
            region = stat["region"]
            sns.kdeplot(stat["df"][variable], label=region, color=region_colors[region])

    # Set title and labels
    plt.title(f"Comparison of {variable_display_name} Across Regions")
    plt.xlabel(f"{variable_display_name} ({units})")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()

    if output_path is not None:
        plt.savefig(Path(output_path, f"{variable}_kde_comparison.png"))
    else:
        plt.show()

    # Create bar charts for percentile metrics
    # Identify percentile metrics (they start with 'p')
    percentile_metrics = [
        m for m in metrics if m.startswith("p") and m not in ["min", "max"]
    ]

    if percentile_metrics:
        # Create a figure with subplots for each percentile
        fig, axes = plt.subplots(
            len(percentile_metrics), 1, figsize=(16, 5 * len(percentile_metrics))
        )

        # Handle case with only one percentile
        if len(percentile_metrics) == 1:
            axes = [axes]

        fig.suptitle(
            f"{variable_display_name} - Percentile Comparison Across Regions ({units})",
            fontsize=16,
        )

        for i, metric in enumerate(sorted(percentile_metrics)):
            # Format metric for display
            if metric.startswith("p"):
                display_metric = metric.replace("p", "") + "%"
                if len(display_metric) > 3:  # For p9999 etc.
                    display_metric = (
                        display_metric[:-2] + "." + display_metric[-2:] + "%"
                    )
            else:
                display_metric = metric

            # Sort data by the current percentile value
            if metric in comparison_table.columns:
                sorted_data = comparison_table.sort_values(by=metric)

                # Create bar plot
                ax = axes[i]
                bars = ax.bar(
                    sorted_data["Region"],
                    sorted_data[metric],
                    color=[region_colors[region] for region in sorted_data["Region"]],
                )

                # Add value labels on top of bars
                for bar in bars:
                    height = bar.get_height()
                    ax.annotate(
                        f"{height:.3f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                    )

                # Set title and labels
                ax.set_title(f"{display_metric} Values", fontsize=14)
                ax.set_xlabel("Region")
                ax.set_ylabel(f"{variable_display_name} ({units})")

                # Rotate x-tick labels for better readability
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for the suptitle
        if output_path is not None:
            plt.savefig(Path(output_path, f"{variable}_bar_comparison.png"))
        else:
            plt.show()

    # Print comparison table
    print(f"\n{variable_display_name} Comparison ({units}):")
    print(comparison_table.to_string(index=False))

    # Return summary
    return {
        "variable": variable,
        "variable_display_name": variable_display_name,
        "regions": regions,
        "comparison_table": comparison_table,
        "region_colors": region_colors,
    }


def copy_images_for_web(
    source_dir, docs_img_dir, regions_processed, max_width=1200, quality=85
):
    """
    Copy and optimize images for web display using PIL/Pillow.

    Args:
        source_dir: Source directory containing original images
        docs_img_dir: Destination directory for web-optimized images
        regions_processed: List of regions to process
        max_width: Maximum width for web images (default 1200px)
        quality: JPEG quality for optimization (default 85)
    """
    try:
        from PIL import Image, ImageOps
        import os

        def optimize_image(src_path, dst_path, max_width=max_width, quality=quality):
            """Optimize a single image for web use."""
            try:
                with Image.open(src_path) as img:
                    # Convert to RGB if necessary (handles RGBA PNGs)
                    if img.mode in ("RGBA", "LA"):
                        # Create white background for transparency
                        background = Image.new("RGB", img.size, (255, 255, 255))
                        if img.mode == "RGBA":
                            background.paste(
                                img, mask=img.split()[-1]
                            )  # Use alpha channel as mask
                        else:
                            background.paste(
                                img, mask=img.split()[-1]
                            )  # Use alpha channel as mask
                        img = background
                    elif img.mode != "RGB":
                        img = img.convert("RGB")

                    # Resize if too wide
                    if img.width > max_width:
                        ratio = max_width / img.width
                        new_height = int(img.height * ratio)
                        img = img.resize(
                            (max_width, new_height), Image.Resampling.LANCZOS
                        )

                    # Auto-orient based on EXIF data
                    img = ImageOps.exif_transpose(img)

                    # Save as optimized PNG or JPEG based on original format
                    if src_path.suffix.lower() == ".png":
                        # For PNG, use optimize flag and reduce colors if possible
                        img.save(dst_path, "PNG", optimize=True)
                    else:
                        # For other formats, save as high-quality JPEG
                        dst_path = dst_path.with_suffix(".jpg")
                        img.save(dst_path, "JPEG", quality=quality, optimize=True)

                    # Get file size reduction info
                    original_size = os.path.getsize(src_path)
                    optimized_size = os.path.getsize(dst_path)
                    reduction = (1 - optimized_size / original_size) * 100

                    print(
                        f"Optimized {src_path.name}: {original_size/1024:.1f}KB → {optimized_size/1024:.1f}KB ({reduction:.1f}% reduction)"
                    )

            except Exception as e:
                print(f"Warning: Could not optimize {src_path.name}: {e}")
                # Fallback to simple copy
                import shutil

                shutil.copy2(src_path, dst_path)

        # Process regional images
        for region in regions_processed:
            region_dir = Path(source_dir, region)
            if region_dir.exists():
                image_files = [
                    f"{region}_mean_sea_water_speed.png",
                    f"{region}_p95_sea_water_speed.png",
                    f"{region}_mean_sea_water_power_density.png",
                    f"{region}_p95_sea_water_power_density.png",
                    f"{region}_distance_to_sea_floor.png",
                ]

                for img_file in image_files:
                    src_path = region_dir / img_file
                    if src_path.exists():
                        dst_path = docs_img_dir / img_file
                        optimize_image(src_path, dst_path)

        # Process comparison images from base directory
        comparison_files = [
            "vap_water_column_mean_sea_water_speed_kde_comparison.png",
            "vap_water_column_95th_percentile_sea_water_speed_kde_comparison.png",
            "vap_water_column_mean_sea_water_power_density_kde_comparison.png",
            "vap_water_column_95th_percentile_sea_water_power_density_kde_comparison.png",
            "vap_sea_floor_depth_kde_comparison.png",
            "vap_water_column_mean_sea_water_speed_bar_comparison.png",
            "vap_water_column_95th_percentile_sea_water_speed_bar_comparison.png",
            "vap_water_column_mean_sea_water_power_density_bar_comparison.png",
            "vap_water_column_95th_percentile_sea_water_power_density_bar_comparison.png",
            "vap_sea_floor_depth_bar_comparison.png",
        ]

        for img_file in comparison_files:
            src_path = Path(source_dir) / img_file
            if src_path.exists():
                dst_path = docs_img_dir / img_file
                optimize_image(src_path, dst_path)

    except ImportError:
        print("Warning: PIL/Pillow not available. Falling back to simple copy.")
        print("Install with: pip install Pillow")

        # Fallback to simple copy
        import shutil

        # Copy regional images
        for region in regions_processed:
            region_dir = Path(source_dir, region)
            if region_dir.exists():
                image_files = [
                    f"{region}_mean_sea_water_speed.png",
                    f"{region}_p95_sea_water_speed.png",
                    f"{region}_mean_sea_water_power_density.png",
                    f"{region}_p95_sea_water_power_density.png",
                    f"{region}_distance_to_sea_floor.png",
                ]

                for img_file in image_files:
                    src_path = region_dir / img_file
                    if src_path.exists():
                        dst_path = docs_img_dir / img_file
                        shutil.copy2(src_path, dst_path)
                        print(f"Copied {img_file} to docs/img/")

        # Copy comparison images
        comparison_files = [
            "vap_water_column_mean_sea_water_speed_kde_comparison.png",
            "vap_water_column_95th_percentile_sea_water_speed_kde_comparison.png",
            "vap_water_column_mean_sea_water_power_density_kde_comparison.png",
            "vap_water_column_95th_percentile_sea_water_power_density_kde_comparison.png",
            "vap_sea_floor_depth_kde_comparison.png",
            "vap_water_column_mean_sea_water_speed_bar_comparison.png",
            "vap_water_column_95th_percentile_sea_water_speed_bar_comparison.png",
            "vap_water_column_mean_sea_water_power_density_bar_comparison.png",
            "vap_water_column_95th_percentile_sea_water_power_density_bar_comparison.png",
            "vap_sea_floor_depth_bar_comparison.png",
        ]

        for img_file in comparison_files:
            src_path = Path(source_dir) / img_file
            if src_path.exists():
                dst_path = docs_img_dir / img_file
                shutil.copy2(src_path, dst_path)
                print(f"Copied {img_file} to docs/img/")


def _print_color_level_ranges(bounds, label, units, cmap, n_colors):
    """
    Print color level ranges and capture data for markdown generation.

    Returns:
        list: Color level data for markdown
    """
    # Capture the data first
    color_data = capture_color_level_ranges(bounds, label, units, cmap, n_colors)

    # Print the existing format (unchanged output)
    print(f"\n{label} Color Level Ranges [{units}]:")
    print("-" * 50)

    for color_info in color_data:
        level_str = f"Level {color_info['level']}: {color_info['range']}"
        print(f"{level_str:<40} | Color: {color_info['hex']}")

    print("-" * 50)

    return color_data


def capture_color_level_ranges(bounds, label, units, cmap, n_colors):
    """
    Capture color level ranges and return structured data for markdown generation.

    Returns:
        list: List of dictionaries containing color level information
    """
    color_levels = []

    # Get the RGB values for each color level
    colors = [cmap(i / (n_colors - 1)) for i in range(n_colors)]

    # Format the RGB values as hex colors and RGB tuples
    for i in range(n_colors):
        if i < n_colors - 1:
            min_val = bounds[i]
            max_val = bounds[i + 1]

            # Get color information
            rgba_color = colors[i]
            rgb_color = rgba_color[:3]  # Remove alpha channel

            # Convert to 0-255 range
            rgb_255 = tuple(int(c * 255) for c in rgb_color)
            hex_color = f"#{rgb_255[0]:02x}{rgb_255[1]:02x}{rgb_255[2]:02x}"

            # Format value range
            max_value = max(abs(min_val), abs(max_val))
            if max_value >= 1000:
                range_str = f"{min_val:.0f} - {max_val:.0f}"
            elif max_value >= 100:
                range_str = f"{min_val:.1f} - {max_val:.1f}"
            elif max_value >= 10:
                range_str = f"{min_val:.2f} - {max_val:.2f}"
            else:
                range_str = f"{min_val:.3f} - {max_val:.3f}"

            color_levels.append(
                {
                    "level": i + 1,
                    "range": f"{range_str} {units}",
                    "min_val": min_val,
                    "max_val": max_val,
                    "hex": hex_color,
                    "rgb": f"rgb({rgb_255[0]}, {rgb_255[1]}, {rgb_255[2]})",
                    "rgb_tuple": rgb_255,
                }
            )
        else:
            # Handle the overflow level (≥ max_value)
            rgba_color = colors[i]
            rgb_color = rgba_color[:3]
            rgb_255 = tuple(int(c * 255) for c in rgb_color)
            hex_color = f"#{rgb_255[0]:02x}{rgb_255[1]:02x}{rgb_255[2]:02x}"

            # Format the overflow range
            max_val = bounds[i - 1] if i > 0 else bounds[0]
            if max_val >= 1000:
                range_str = f"≥ {max_val:.0f}"
            elif max_val >= 100:
                range_str = f"≥ {max_val:.1f}"
            elif max_val >= 10:
                range_str = f"≥ {max_val:.2f}"
            else:
                range_str = f"≥ {max_val:.3f}"

            color_levels.append(
                {
                    "level": i + 1,
                    "range": f"{range_str} {units}",
                    "min_val": max_val,
                    "max_val": float("inf"),
                    "hex": hex_color,
                    "rgb": f"rgb({rgb_255[0]}, {rgb_255[1]}, {rgb_255[2]})",
                    "rgb_tuple": rgb_255,
                }
            )

    return color_levels


# Modified color printing function that also captures data
def _print_and_capture_color_level_ranges(bounds, label, units, cmap, n_colors):
    """
    Print color level ranges and capture data for markdown generation.

    Returns:
        list: Color level data for markdown
    """
    # Capture the data first
    color_data = capture_color_level_ranges(bounds, label, units, cmap, n_colors)

    # Print the existing format
    print(f"\n{label} Color Level Ranges [{units}]:")
    print("-" * 50)

    for color_info in color_data:
        level_str = f"Level {color_info['level']}: {color_info['range']}"
        print(f"{level_str:<40} | Color: {color_info['hex']}")

    print("-" * 50)

    return color_data


def generate_markdown_specification(
    regions_processed,
    output_dir,
    mean_speed_summary,
    max_speed_summary,
    mean_power_density_summary,
    max_power_density_summary,
    sea_floor_depth_summary,
    color_level_data=None,
):
    """
    Generate a markdown specification file documenting all visualizations.

    Args:
        regions_processed: List of region names that were processed
        output_dir: Base output directory path
        *_summary: Summary objects from analyze_variable_across_regions
        color_level_data: Dictionary containing color level information for each variable
    """

    # Create docs/img directory for web-sized images
    docs_img_dir = Path("docs/img")
    docs_img_dir.mkdir(parents=True, exist_ok=True)

    # Copy images to docs/img directory
    copy_images_for_web(output_dir, docs_img_dir, regions_processed)

    # Markdown content
    md_content = []

    # Header
    md_content.extend(
        [
            "# ME Atlas High Resolution Tidal Data QOI Visualization Specification",
            "",
            "This document provides a comprehensive specification of all visualizations generated from the high-resolution tidal data analysis.",
            "",
            f"**Generated on:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Regions processed:** {len(regions_processed)}",
            "",
            "## Overview",
            "",
            "The visualization suite includes the following variable types:",
            "",
            "| Variable | Units | Description |",
            "| -------- | ----- | ----------- |",
            "| Mean Sea Water Speed | m/s | Time-averaged current velocity magnitude |",
            "| 95th Percentile Sea Water Speed | m/s | High-energy current events (top 5% of observations) |",
            "| Mean Sea Water Power Density | W/m² | Time-averaged kinetic energy flux |",
            "| 95th Percentile Sea Water Power Density | W/m² | High-energy power density events |",
            "| Sea Floor Depth | m | Distance from surface to sea floor (where available) |",
            "",
            "Each variable is visualized both as spatial maps and statistical distributions.",
            "",
        ]
    )

    # Enhanced visualization specifications with color details
    viz_specs = {
        "mean_sea_water_speed": {
            "title": "Mean Sea Water Speed",
            "units": "m/s",
            "column_name": "vap_water_column_mean_sea_water_speed",
            "colormap": "cmocean.thermal",
            "range_min": SEA_WATER_SPEED_CBAR_MIN,
            "range_max": SEA_WATER_SPEED_CBAR_MAX,
            "levels": SEA_WATER_SPEED_LEVELS,
            "physical_meaning": "Yearly average of depth averaged current velocity magnitude",
            "intended_usage": "Estimation of tidal turbine power generation potential",
        },
        "p95_sea_water_speed": {
            "title": "95th Percentile Sea Water Speed",
            "units": "m/s",
            "column_name": "vap_water_column_95th_percentile_sea_water_speed",
            "colormap": "cmocean.matter",
            "range_min": SEA_WATER_SPEED_CBAR_MIN,
            "range_max": SEA_WATER_MAX_SPEED_CBAR_MAX,
            "levels": SEA_WATER_MAX_SPEED_LEVELS,
            "physical_meaning": "95th percentile of yearly depth maximum current velocity magnitude",
            "intended_usage": "Estimation of severity of extreme tidal events",
        },
        "mean_sea_water_power_density": {
            "title": "Mean Sea Water Power Density",
            "units": "W/m²",
            "column_name": "vap_water_column_mean_sea_water_power_density",
            "colormap": "cmocean.dense",
            "range_min": SEA_WATER_POWER_DENSITY_CBAR_MIN,
            "range_max": SEA_WATER_POWER_DENSITY_CBAR_MAX,
            "levels": SEA_WATER_POWER_DENSITY_LEVELS,
            "physical_meaning": "Yearly average of depth averaged power density (kinetic energy flux)",
        },
        "p95_sea_water_power_density": {
            "title": "95th Percentile Sea Water Power Density",
            "units": "W/m²",
            "column_name": "vap_water_column_95th_percentile_sea_water_power_density",
            "colormap": "cmocean.amp",
            "range_min": SEA_WATER_POWER_DENSITY_CBAR_MIN,
            "range_max": SEA_WATER_MAX_POWER_DENSITY_CBAR_MAX,
            "levels": SEA_WATER_MAX_POWER_DENSITY_LEVELS,
            "physical_meaning": "95th percentile of the yearly maximum of depth averaged power density (kinetic energy flux)",
        },
        "distance_to_sea_floor": {
            "title": "Distance to Sea Floor",
            "units": "m",
            "column_name": "vap_sea_floor_depth",
            "colormap": "cmocean.deep",
            "range_min": SEA_FLOOR_DEPTH_MIN,
            "range_max": SEA_FLOOR_DEPTH_MAX,
            "levels": SEA_FLOOR_DEPTH_LEVELS,
            "physical_meaning": "Yearly average distance from the water surface to the sea floor",
        },
    }

    # Add detailed specifications section
    md_content.extend(
        [
            "## Visualization Specifications",
            "",
            "### Variable Details",
            "",
            "| Variable | Column Name | Range | Units | Discrete Levels | Colormap |",
            "| -------- | ----------- | ----- | ----- | --------------- | -------- |",
        ]
    )

    for var_key, spec in viz_specs.items():
        range_str = f"{spec['range_min']} - {spec['range_max']}"
        md_content.append(
            f"| {spec['title']} | `{spec['column_name']}` | {range_str} | {spec['units']} | {spec['levels']} | {spec['colormap']} |"
        )

    md_content.extend(
        [
            "",
            "### Physical Interpretation",
            "",
            "| Variable | Physical Meaning |",
            "| -------- | ---------------- |",
        ]
    )

    for var_key, spec in viz_specs.items():
        md_content.append(f"| {spec['title']} | {spec['physical_meaning']} |")

    md_content.append("")

    # Add color mapping details if available
    if color_level_data:
        md_content.extend(
            [
                "## Color Mapping Specifications",
                "",
                "**The following tables provide exact color specifications for each variable.",
                "All colors use discrete levels with an overflow level for values exceeding the maximum range.",
                "",
            ]
        )

        for var_key, spec in viz_specs.items():
            if var_key in color_level_data:
                md_content.extend(
                    [
                        f"### {spec['title']} ({spec['units']})",
                        "",
                        f"**Colormap:** {spec['colormap']}",
                        f"**Data Range:** {spec['range_min']} to {spec['range_max']} {spec['units']}",
                        f"**Discrete Levels:** {spec['levels']} within range + 1 overflow level",
                        "",
                        "| Level | Value Range | Hex Color | RGB Color | Color Preview |",
                        "| ----- | ----------- | --------- | --------- | ------------- |",
                    ]
                )

                # Add color level data here (this will be populated when we capture the print output)
                colors_info = color_level_data[var_key]
                for i, color_info in enumerate(colors_info):
                    level_num = i + 1
                    value_range = color_info["range"]
                    hex_color = color_info["hex"]
                    rgb_color = color_info["rgb"]
                    # Create a small color block using HTML
                    color_preview = f'<span style="background-color:{hex_color}; color:white; padding:2px 8px; border-radius:3px;">████</span>'
                    md_content.append(
                        f"| {level_num} | {value_range} | `{hex_color}` | `{rgb_color}` | {color_preview} |"
                    )

                md_content.append("")
    else:
        md_content.extend(
            [
                "## Color Mapping Specifications",
                "",
                "**Note:** Color level details will be populated when the visualization functions are executed.",
                "The color mapping system uses discrete levels for improved interpretation.",
                "",
            ]
        )

    # Regional visualizations section with images
    md_content.extend(
        [
            "## Regional Visualizations",
            "",
            "The following regions have been processed with complete visualization suites.",
            "All images are optimized for web display and stored in the `docs/img/` directory.",
            "",
        ]
    )

    for region in regions_processed:
        region_title = region.replace("_", " ").title()
        md_content.extend(
            [
                f"### {region_title}",
                "",
                "#### Spatial Distribution Maps",
                "",
            ]
        )

        # Add each visualization with image embed and caption
        viz_types = [
            ("mean_sea_water_speed", "Mean Sea Water Speed"),
            ("p95_sea_water_speed", "95th Percentile Sea Water Speed"),
            ("mean_sea_water_power_density", "Mean Sea Water Power Density"),
            ("p95_sea_water_power_density", "95th Percentile Sea Water Power Density"),
            ("distance_to_sea_floor", "Distance to Sea Floor (if available)"),
        ]

        for viz_key, viz_title in viz_types:
            img_filename = f"{region}_{viz_key}.png"
            img_path = f"docs/img/{img_filename}"

            # Get units from viz_specs
            units = next(
                (
                    spec["units"]
                    for spec in viz_specs.values()
                    if viz_title.startswith(spec["title"].split()[0])
                ),
                "",
            )

            md_content.extend(
                [
                    f"**{viz_title}**",
                    "",
                    f"![{viz_title} for {region_title}]({img_path})",
                    f"*Figure: {viz_title} spatial distribution for {region_title}. Units: {units}. Discrete color levels enhance interpretation for decision-making applications.*",
                    "",
                ]
            )

        md_content.extend(
            [
                "#### File Paths",
                "",
                "| Visualization Type | File Path | Units |",
                "| ------------------ | --------- | ----- |",
            ]
        )

        for viz_key, viz_title in viz_types:
            img_filename = f"{region}_{viz_key}.png"
            img_path = f"docs/img/{img_filename}"
            units = next(
                (
                    spec["units"]
                    for spec in viz_specs.values()
                    if viz_title.startswith(spec["title"].split()[0])
                ),
                "",
            )
            md_content.append(f"| {viz_title} | `{img_path}` | {units} |")

        md_content.extend(["", "---", ""])

    # Cross-regional analysis section with images
    md_content.extend(
        [
            "## Cross-Regional Comparative Analysis",
            "",
            "Comparative visualizations across all processed regions provide insights into spatial variability and patterns.",
            "",
            "### Probability Density Function (KDE) Comparisons",
            "",
            "These plots show the statistical distribution of each variable across different regions.",
            "",
        ]
    )

    kde_comparisons = [
        (
            "vap_water_column_mean_sea_water_speed_kde_comparison.png",
            "Mean Sea Water Speed",
            "m/s",
        ),
        (
            "vap_water_column_95th_percentile_sea_water_speed_kde_comparison.png",
            "95th Percentile Sea Water Speed",
            "m/s",
        ),
        (
            "vap_water_column_mean_sea_water_power_density_kde_comparison.png",
            "Mean Sea Water Power Density",
            "W/m²",
        ),
        (
            "vap_water_column_95th_percentile_sea_water_power_density_kde_comparison.png",
            "95th Percentile Sea Water Power Density",
            "W/m²",
        ),
        ("vap_sea_floor_depth_kde_comparison.png", "Sea Floor Depth", "m"),
    ]

    for img_file, title, units in kde_comparisons:
        img_path = f"docs/img/{img_file}"
        md_content.extend(
            [
                f"**{title} Distribution Comparison**",
                "",
                f"![{title} KDE Comparison]({img_path})",
                f"*Figure: Kernel Density Estimation comparison of {title.lower()} across all processed regions. Units: {units}. Shows probability density functions for statistical comparison.*",
                "",
            ]
        )

    md_content.extend(
        [
            "### Percentile Bar Chart Comparisons",
            "",
            "These charts compare key percentile values across regions for quantitative analysis.",
            "",
        ]
    )

    bar_comparisons = [
        (
            "vap_water_column_mean_sea_water_speed_bar_comparison.png",
            "Mean Sea Water Speed",
            "m/s",
        ),
        (
            "vap_water_column_95th_percentile_sea_water_speed_bar_comparison.png",
            "95th Percentile Sea Water Speed",
            "m/s",
        ),
        (
            "vap_water_column_mean_sea_water_power_density_bar_comparison.png",
            "Mean Sea Water Power Density",
            "W/m²",
        ),
        (
            "vap_water_column_95th_percentile_sea_water_power_density_bar_comparison.png",
            "95th Percentile Sea Water Power Density",
            "W/m²",
        ),
        ("vap_sea_floor_depth_bar_comparison.png", "Sea Floor Depth", "m"),
    ]

    for img_file, title, units in bar_comparisons:
        img_path = f"docs/img/{img_file}"
        md_content.extend(
            [
                f"**{title} Percentile Comparison**",
                "",
                f"![{title} Bar Comparison]({img_path})",
                f"*Figure: Percentile values of {title.lower()} compared across all processed regions. Units: {units}. Enables quantitative comparison of regional characteristics.*",
                "",
            ]
        )

    # Technical implementation details
    md_content.extend(
        [
            "## Technical Implementation Details",
            "",
            "### Color Mapping System",
            "",
            "| Parameter | Specification |",
            "| --------- | ------------- |",
            "| Color Levels | Discrete levels within specified range + 1 overflow level |",
            "| Normalization | BoundaryNorm with n_colors + 1 boundaries |",
            "| Overflow Handling | Values > max_range mapped to final color level |",
            "| Color Source | cmocean scientific colormaps for perceptual uniformity |",
            "",
            "### Projection Systems",
            "",
            "| Region Type | Projection | EPSG Code | Usage |",
            "| ----------- | ---------- | --------- | ----- |",
            "| Aleutian | Orthographic | Custom | Polar regions crossing 180° meridian |",
            "| Standard | Web Mercator | EPSG:3857 | All other coastal regions |",
            "",
            "### Data Processing Pipeline",
            "",
            "| Step | Description |",
            "| ---- | ----------- |",
            "| 1. Data Loading | Read parquet files from regional directories |",
            "| 2. Mesh Detection | Check for element corner data availability |",
            "| 3. Projection Setup | Choose appropriate coordinate system |",
            "| 4. Color Mapping | Apply discrete colormap with boundary normalization |",
            "| 5. Rendering | Triangulated mesh or point scatter plot |",
            "| 6. Overlay | Add basemap, coastlines, and geographic features |",
            "",
            "### File Organization",
            "",
            "| Directory | Contents | Purpose |",
            "| --------- | -------- | ------- |",
            "| `docs/img/` | Web-optimized images (PNG, max 1200px width) | Documentation and web display |",
            "| `{region}/` | Full-resolution images (PNG, 300 DPI) | Production and archival |",
            "| Root | Cross-regional comparison plots | Statistical analysis |",
            "",
            "### Image Optimization",
            "",
            "| Parameter | Value | Purpose |",
            "| --------- | ----- | ------- |",
            "| Maximum Width | 1200 pixels | Web performance optimization |",
            "| Format | PNG (optimized) | Lossless compression with transparency support |",
            "| Background | White | Transparency conversion for web compatibility |",
            "| Resampling | LANCZOS | High-quality image resizing |",
            "",
        ]
    )

    # Summary statistics (if available)
    if mean_speed_summary and "comparison_table" in mean_speed_summary:
        md_content.extend(
            [
                "## Regional Comparison Summary",
                "",
                "Statistical summaries and percentile comparisons are documented in the cross-regional analysis visualizations.",
                "Key insights and quantitative comparisons are available in the generated bar charts and KDE plots.",
                "",
            ]
        )

    # Implementation guidance
    md_content.extend(
        [
            "## Implementation Guidance",
            "",
            "### For Development Teams",
            "",
            "1. **Color Implementation**: Use the exact hex codes provided in the color mapping tables",
            "2. **Value Ranges**: Implement discrete boundaries as specified for each variable",
            "3. **Overflow Handling**: Values exceeding maximum range should use the final color level",
            "4. **Units**: Always display units as specified in the variable tables",
            "5. **Projections**: Use appropriate coordinate systems based on region type",
            "",
            "### Quality Assurance",
            "",
            "- Verify color accuracy using the provided hex codes",
            "- Test overflow handling with extreme values",
            "- Validate unit display and scientific notation",
            "- Check projection accuracy for cross-meridian regions",
            "",
            "### Performance Considerations",
            "",
            "- Use optimized images from `docs/img/` for web applications",
            "- Consider progressive loading for large datasets",
            "- Implement client-side caching for repeated access",
            "",
        ]
    )

    # Footer
    md_content.extend(
        [
            "---",
            "",
            "## Document Information",
            "",
            f"- **Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S UTC')}",
            f"- **Regions Processed:** {', '.join(regions_processed)}",
            f"- **Total Visualizations:** {len(regions_processed) * 5 + len(kde_comparisons) + len(bar_comparisons)}",
            "- **Color System:** Discrete levels with scientific colormaps",
            "- **Coordinate Systems:** Orthographic (Aleutian) / Web Mercator (Standard)",
            "",
            "*This specification was auto-generated from the tidal data visualization pipeline.*",
            "*All color codes, ranges, and technical specifications are programmatically derived.*",
        ]
    )

    # Write the markdown file
    output_file = Path(
        "me_atlas_high_resolution_tidal_data_qoi_visualization_specification.md"
    )
    with open(output_file, "w") as f:
        f.write("\n".join(md_content))

    print(f"Markdown specification written to: {output_file}")
    return output_file


if __name__ == "__main__":
    # Display available regions
    regions = get_available_regions()
    regions.reverse()
    print("Available regions:")
    for i, region in enumerate(regions):
        print(f"{i+1}. {region}")

    mean_speed_stats = []
    max_speed_stats = []
    mean_power_density_stats = []
    max_power_density_stats = []
    sea_floor_depth_stats = []

    color_level_data = {}

    for i in range(0, 5):
        # Get the parquet file path
        selected_region = regions[i]
        parquet_file = get_parquet_path(selected_region)
        print(f"Reading file: {parquet_file}")

        # Read the parquet file
        df = pd.read_parquet(parquet_file)

        this_output_path = Path(VIZ_OUTPUT_DIR, selected_region)
        this_output_path.mkdir(parents=True, exist_ok=True)

        print(f"\tPlotting {selected_region} mean_sea_water_speed...")

        mean_speed_stats.append(
            analyze_variable(
                df,
                "vap_water_column_mean_sea_water_speed",
                "Mean Sea Water Speed",
                selected_region,
                output_path=Path(this_output_path),
            )
        )

        fig, ax, color_data = plot_tidal_variable(
            df,
            selected_region,
            "vap_water_column_mean_sea_water_speed",
            "Mean Sea Water Speed",
            SEA_WATER_SPEED_UNITS,
            SEA_WATER_SPEED_CBAR_MIN,
            SEA_WATER_SPEED_CBAR_MAX,
            is_aleutian="aleutian" in selected_region,
            cmap=MEAN_SPEED_CMAP,
            save_path=Path(
                this_output_path,
                f"{selected_region}_mean_sea_water_speed.png",
            ),
            n_colors=SEA_WATER_SPEED_LEVELS,
        )

        if "mean_sea_water_speed" not in color_level_data:
            color_level_data["mean_sea_water_speed"] = color_data

        plt.close()
        print(f"\tPlotting {selected_region} p95_sea_water_speed...")

        # Capture color level data for markdown generation

        max_speed_stats.append(
            analyze_variable(
                df,
                "vap_water_column_95th_percentile_sea_water_speed",
                "95th Percentile Sea Water Speed",
                selected_region,
                output_path=Path(this_output_path),
            )
        )

        fig, ax, color_data = plot_tidal_variable(
            df,
            selected_region,
            "vap_water_column_95th_percentile_sea_water_speed",
            "95th Percentile Sea Water Speed",
            SEA_WATER_SPEED_UNITS,
            SEA_WATER_SPEED_CBAR_MIN,
            SEA_WATER_MAX_SPEED_CBAR_MAX,
            is_aleutian="aleutian" in selected_region,
            cmap=MAX_SPEED_CMAP,
            save_path=Path(
                this_output_path,
                f"{selected_region}_p95_sea_water_speed.png",
            ),
            n_colors=SEA_WATER_MAX_SPEED_LEVELS,
        )

        if "p95_sea_water_speed" not in color_level_data:
            color_level_data["p95_sea_water_speed"] = color_data

        plt.close()

        print(f"\tPlotting {selected_region} mean_sea_water_power_density...")

        mean_power_density_stats.append(
            analyze_variable(
                df,
                "vap_water_column_mean_sea_water_power_density",
                "Mean Sea Water Power Density",
                selected_region,
                output_path=Path(this_output_path),
            )
        )

        fig, ax, color_data = plot_tidal_variable(
            df,
            selected_region,
            "vap_water_column_mean_sea_water_power_density",
            "Mean Sea Water Power Density",
            SEA_WATER_POWER_DENSITY_UNITS,
            SEA_WATER_POWER_DENSITY_CBAR_MIN,
            SEA_WATER_POWER_DENSITY_CBAR_MAX,
            is_aleutian="aleutian" in selected_region,
            cmap=MEAN_POWER_DENSITY_CMAP,
            save_path=Path(
                this_output_path,
                f"{selected_region}_mean_sea_water_power_density.png",
            ),
            n_colors=SEA_WATER_POWER_DENSITY_LEVELS,
        )

        if "mean_sea_water_power_density" not in color_level_data:
            color_level_data["mean_sea_water_power_density"] = color_data

        plt.close()

        print(f"\tPlotting {selected_region} p95_sea_water_power_density...")

        max_power_density_stats.append(
            analyze_variable(
                df,
                "vap_water_column_95th_percentile_sea_water_power_density",
                "95th Percentile Sea Water Power Density",
                selected_region,
                output_path=Path(this_output_path),
            )
        )

        fig, ax, color_data = plot_tidal_variable(
            df,
            selected_region,
            "vap_water_column_95th_percentile_sea_water_power_density",
            "Max Sea Water Power Density",
            SEA_WATER_POWER_DENSITY_UNITS,
            SEA_WATER_POWER_DENSITY_CBAR_MIN,
            SEA_WATER_MAX_POWER_DENSITY_CBAR_MAX,
            is_aleutian="aleutian" in selected_region,
            cmap=MAX_POWER_DENSITY_CMAP,
            save_path=Path(
                this_output_path,
                f"{selected_region}_p95_sea_water_power_density.png",
            ),
            n_colors=SEA_WATER_MAX_POWER_DENSITY_LEVELS,
        )

        if "p95_sea_water_power_density" not in color_level_data:
            color_level_data["p95_sea_water_power_density"] = color_data

        plt.close()
        if "vap_sea_floor_depth" in df.columns:
            sea_floor_depth_stats.append(
                analyze_variable(
                    df,
                    "vap_sea_floor_depth",
                    "Sea Floor Depth",
                    selected_region,
                    output_path=Path(this_output_path),
                )
            )
            fig, ax, color_data = plot_tidal_variable(
                df,
                selected_region,
                "vap_sea_floor_depth",
                "Distance to Sea Floor",
                "m",
                SEA_FLOOR_DEPTH_MIN,
                SEA_FLOOR_DEPTH_MAX,
                is_aleutian="aleutian" in selected_region,
                cmap=SEA_FLOOR_DEPTH_CMAP,
                save_path=Path(
                    this_output_path,
                    f"{selected_region}_distance_to_sea_floor.png",
                ),
                n_colors=SEA_FLOOR_DEPTH_LEVELS,
            )

            if "sea_floor_depth" not in color_level_data:
                color_level_data["sea_floor_depth"] = color_data

            plt.close()

    print("Calculating and Plotting Speed variable_summary...")
    mean_speed_summary = analyze_variable_across_regions(
        mean_speed_stats, output_path=VIZ_OUTPUT_DIR
    )
    max_speed_summary = analyze_variable_across_regions(
        max_speed_stats, output_path=VIZ_OUTPUT_DIR
    )
    mean_power_density_summary = analyze_variable_across_regions(
        mean_power_density_stats, output_path=VIZ_OUTPUT_DIR
    )
    max_power_density_summary = analyze_variable_across_regions(
        max_power_density_stats, output_path=VIZ_OUTPUT_DIR
    )
    sea_floor_depth_summary = analyze_variable_across_regions(
        sea_floor_depth_stats, output_path=VIZ_OUTPUT_DIR
    )

    # After all the plotting and analysis is complete, add:
    print("Generating markdown specification...")

    # Generate the markdown specification
    generate_markdown_specification(
        regions_processed=regions,
        output_dir=VIZ_OUTPUT_DIR,
        mean_speed_summary=mean_speed_summary,
        max_speed_summary=max_speed_summary,
        mean_power_density_summary=mean_power_density_summary,
        max_power_density_summary=max_power_density_summary,
        sea_floor_depth_summary=sea_floor_depth_summary,
        color_level_data=color_level_data,
    )

    print("Analysis and documentation complete!")
