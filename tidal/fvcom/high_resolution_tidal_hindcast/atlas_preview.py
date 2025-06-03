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

from config import config


# Set the base directory - modify this to match your system
BASE_DIR = Path("/projects/hindcastra/Tidal/datasets/high_resolution_tidal_hindcast")

VIZ_OUTPUT_DIR = Path("/home/asimms/tidal/analysis/viz/")
VIZ_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEA_WATER_SPEED_CBAR_MAX = 1.5
SEA_WATER_SPEED_CBAR_MIN = 0.0
# In the output visualization, there will be 9 levels, but this is the number of levels within the range
# The 9th level is for values outside the range.
SEA_WATER_SPEED_LEVELS = 10

SEA_WATER_MAX_SPEED_CBAR_MAX = 4.0
SEA_WATER_MAX_SPEED_LEVELS = 8

# SEA_WATER_POWER_DENSITY_CBAR_MAX = 4000  # 0.5 * 1025 * (2.0^3) = 4,100
SEA_WATER_POWER_DENSITY_CBAR_MAX = 1750  # 0.5 * 1025 * (1.5^3) = 1,729.6875
SEA_WATER_POWER_DENSITY_CBAR_MIN = 0
# SEA_WATER_POWER_DENSITY_LEVELS = 8
SEA_WATER_POWER_DENSITY_LEVELS = 7

# SEA_WATER_MAX_POWER_DENSITY_CBAR_MAX = 64000  # 0.5 * 1025 * (5.0^3) = 64,062.5
SEA_WATER_MAX_POWER_DENSITY_CBAR_MAX = 32000  # 0.5 * 1025 * (4.0^3) = 32,800
SEA_WATER_MAX_POWER_DENSITY_LEVELS = 8

SEA_FLOOR_DEPTH_MIN = 0
SEA_FLOOR_DEPTH_MAX = 200
SEA_FLOOR_DEPTH_LEVELS = 10

BASEMAP_PROVIDER = ctx.providers.Esri.WorldImagery

# SEA_WATER_SPEED_UNITS = r"$m/s$"
SEA_WATER_SPEED_UNITS = "m/s"
# SEA_WATER_POWER_DENSITY_UNITS = r"$W/m^2$"
SEA_WATER_POWER_DENSITY_UNITS = "W/m^2"

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
            tick_format = "%.0f"
        elif max_value >= 10:
            tick_format = "%.2f"
        else:
            tick_format = "%.2f"

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
    region_stats: List[Dict[str, Any]],
    output_path: Union[Path, None] = None,
    viz_max: float = None,
) -> Dict[str, Any]:
    """
    Perform meta-analysis with visual justification of viz_max choice.

    Creates a comprehensive summary visualization showing:
    1. Full data distribution with viz_max overlay
    2. Data retention statistics
    3. Regional comparison within viz_max bounds
    """
    if not region_stats:
        print("No region statistics provided for analysis.")
        return {}

    # Extract variable info and standardize units
    variable = region_stats[0]["variable"]
    variable_display_name = region_stats[0]["variable_display_name"]
    units = standardize_units(variable, region_stats[0].get("units", ""))
    regions = [stat["region"] for stat in region_stats if stat["region"] is not None]

    # Collect all data and regional data
    all_data = []
    regional_data = {}

    for stat in region_stats:
        if stat["region"] is not None and "df" in stat:
            region_data = stat["df"][variable].values
            all_data.extend(region_data)
            regional_data[stat["region"]] = region_data

    if not all_data:
        print("No data found for visualization.")
        return {}

    all_data = np.array(all_data)

    # Create the main justification visualization
    if viz_max is not None:
        create_viz_max_justification_plot(
            all_data,
            regional_data,
            variable_display_name,
            units,
            viz_max,
            output_path,
            variable,
        )

    # Create standard comparison visualizations (existing functionality)
    create_standard_comparison_plots(
        region_stats, variable, variable_display_name, units, regions, output_path
    )

    # Return results with validation metrics
    return compile_results(
        variable, variable_display_name, units, regions, region_stats, all_data, viz_max
    )


def create_viz_max_justification_plot(
    all_data, regional_data, var_name, units, viz_max, output_path, variable
):
    """Create an improved viz_max justification visualization using seaborn"""

    # Set palette
    palette = sns.color_palette("deep", len(regional_data))

    # Calculate key statistics
    total_points = len(all_data)
    retained_points = np.sum(all_data <= viz_max)
    retention_rate = (retained_points / total_points) * 100

    p50, p90, p95, p99, p999 = np.percentile(all_data, [50, 90, 95, 99, 99.9])
    data_max = np.max(all_data)

    # Create figure with improved layout
    fig = plt.figure(figsize=(24, 14))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.2, 2, 0.8], width_ratios=[2, 1.5, 1.5])

    # Main title
    fig.suptitle(
        f"Colorbar Maximum Justification: {var_name}",
        fontsize=22,
        fontweight="bold",
        y=0.99,
    )

    # 1. Regional histograms with proper separation (top spanning)
    ax1 = fig.add_subplot(gs[0, :2])

    # Create stacked histogram data for regions
    region_names = list(regional_data.keys())
    region_colors = dict(zip(region_names, palette))

    # Calculate optimal bins for all data
    n_bins = min(60, int(np.sqrt(total_points)))
    bin_range = (0, min(viz_max * 1.15, data_max * 1.05))

    # Create separate histograms for each region
    hist_data = []
    labels = []
    colors = []

    for region, data in regional_data.items():
        # Filter data to reasonable range for visualization
        filtered_data = data[data <= bin_range[1]]
        hist_data.append(filtered_data)
        labels.append(f"{region} (n={len(data):,})")
        colors.append(region_colors[region])

    # Create stacked histogram
    ax1.hist(
        hist_data,
        bins=n_bins,
        range=bin_range,
        alpha=0.5,
        label=labels,
        color=colors,
        stacked=True,
        density=False,
    )

    # Add viz_max line
    ax1.axvline(
        viz_max,
        color="red",
        linewidth=4,
        linestyle="-",
        label=f"Viz Max: {viz_max:.1f} {units}",
        zorder=10,
    )

    # Add subtle percentile lines
    ax1.axvline(
        p99,
        color="orange",
        linewidth=2,
        linestyle="--",
        alpha=0.8,
        label=f"99th %ile: {p99:.1f}",
        zorder=9,
    )
    ax1.axvline(
        p95,
        color="gray",
        linewidth=1.5,
        linestyle=":",
        alpha=0.7,
        label=f"95th %ile: {p95:.1f}",
        zorder=8,
    )

    ax1.set_title(
        f"Regional Data Distributions (Total: n={total_points:,} points)",
        fontsize=16,
        pad=15,
    )
    ax1.set_xlabel(f"{var_name} ({units})", fontsize=12)
    ax1.set_ylabel("Frequency", fontsize=12)
    ax1.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=10)

    # Add retention rate annotation
    ax1.text(
        0.35,
        0.85,
        f"Data Retained: {retention_rate:.2f}%\n({retained_points:,}/{total_points:,} points)",
        transform=ax1.transAxes,
        fontsize=12,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="lightgreen", alpha=0.9),
    )

    # 2. Clean KDE plot for regional comparison (main central plot)
    ax2 = fig.add_subplot(gs[1, :2])

    # Create combined dataframe for seaborn
    plot_data = []
    for region, data in regional_data.items():
        # Filter to visualization range for clean comparison
        filtered_data = data[data <= viz_max * 1.1]
        for value in filtered_data:
            plot_data.append({"Region": region, "Value": value, "Count": len(data)})

    plot_df = pd.DataFrame(plot_data)

    # Create KDE plot with seaborn
    for i, (region, data) in enumerate(regional_data.items()):
        filtered_data = data[data <= viz_max * 1.1]
        if len(filtered_data) > 0:
            sns.kdeplot(
                data=filtered_data,
                label=f"{region} (n={len(data):,})",
                color=region_colors[region],
                linewidth=2.5,
                ax=ax2,
            )

    # Add reference lines
    ax2.axvline(
        viz_max,
        color="red",
        linewidth=4,
        linestyle="-",
        label=f"Viz Max: {viz_max:.1f}",
        zorder=10,
    )
    ax2.axvline(
        p99,
        color="orange",
        linewidth=2,
        linestyle="--",
        alpha=0.8,
        label=f"99th %ile: {p99:.1f}",
        zorder=9,
    )
    ax2.axvline(
        p95,
        color="gray",
        linewidth=1.5,
        linestyle=":",
        alpha=0.7,
        label=f"95th %ile: {p95:.1f}",
        zorder=8,
    )

    ax2.set_title(
        "Regional Distribution Comparison (Density View)", fontsize=16, pad=15
    )
    ax2.set_xlabel(f"{var_name} ({units})", fontsize=12)
    ax2.set_ylabel("Density", fontsize=12)
    ax2.set_xlim(0, viz_max * 1.1)
    ax2.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=10)

    # 3. Statistics table (right side, spanning two rows)
    ax3 = fig.add_subplot(gs[:2, 2])
    ax3.axis("off")

    # Create enhanced statistics table
    stats_data = [
        ["Metric", "Value", "% of Viz Max"],
        ["99th percentile", f"{p99:.2f}", f"{(p99/viz_max)*100:.2f}%"],
        ["99.9th percentile", f"{p999:.2f}", f"{(p999/viz_max)*100:.2f}%"],
        ["Maximum", f"{data_max:.2f}", f"{(data_max/viz_max)*100:.2f}%"],
        ["", "", ""],
        ["Points Retained", f"{retained_points:,}", f"{retention_rate:.2f}%"],
        [
            "Points Filtered",
            f"{total_points-retained_points:,}",
            f"{100-retention_rate:.1f}%",
        ],
    ]

    # Create table
    table = ax3.table(
        cellText=stats_data[1:],
        colLabels=stats_data[0],
        cellLoc="center",
        loc="upper center",
        bbox=[0, 0.2, 1, 0.8],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.8)

    # Style the table
    table[(5, 0)].set_facecolor("#90EE90")  # Light green for retained
    table[(5, 1)].set_facecolor("#90EE90")
    table[(5, 2)].set_facecolor("#90EE90")
    table[(6, 0)].set_facecolor("#FFE4B5")  # Light orange for filtered
    table[(6, 1)].set_facecolor("#FFE4B5")
    table[(6, 2)].set_facecolor("#FFE4B5")

    ax3.set_title("Statistics", fontsize=16, pad=20, fontweight="bold")

    # 4. Improved outlier visualization (bottom)
    ax4 = fig.add_subplot(gs[2, :2])

    outliers = all_data[all_data > viz_max]
    if len(outliers) > 0:
        # Create a proper histogram for outliers
        outlier_range = (viz_max, min(data_max * 1.02, viz_max * 2))
        n_outlier_bins = min(30, max(5, len(outliers) // 10))

        counts, bins, patches = ax4.hist(
            outliers,
            bins=n_outlier_bins,
            range=outlier_range,
            alpha=0.7,
            color="lightcoral",
            edgecolor="darkred",
            linewidth=0.5,
        )

        # Add viz_max reference line
        ax4.axvline(
            viz_max,
            color="red",
            linewidth=4,
            linestyle="-",
            label=f"Viz Max Cutoff: {viz_max:.1f}",
            zorder=10,
        )

        ax4.set_xlabel(f"{var_name} ({units})", fontsize=12)
        ax4.set_ylabel("Frequency", fontsize=12)
        ax4.set_title(
            f"Filtered Outliers: {len(outliers):,} points ({((len(outliers)/total_points)*100):.2f}% of data)",
            fontsize=14,
            pad=10,
        )
        ax4.legend(fontsize=10)

        # Add summary statistics for outliers
        if len(outliers) > 1:
            outlier_stats = f"Outlier Range: {np.min(outliers):.2f} to {np.max(outliers):.2f} {units}\nMean: {np.mean(outliers):.2f} {units}"
        else:
            outlier_stats = f"Single outlier: {outliers[0]:.2f} {units}"

        ax4.text(
            0.50,
            0.85,
            outlier_stats,
            transform=ax4.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.9),
            fontsize=10,
        )

    else:
        # No outliers case
        ax4.text(
            0.5,
            0.5,
            "✓ No outliers beyond visualization maximum\nAll data retained for analysis",
            transform=ax4.transAxes,
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.9),
        )
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.set_xticks([])
        ax4.set_yticks([])
        ax4.set_title("Outlier Assessment", fontsize=14, pad=10)

    # Add assessment text in bottom right
    ax5 = fig.add_subplot(gs[2, 2])
    ax5.axis("off")

    # Generate assessment
    if retention_rate >= 99.5:
        assessment = f"✓ EXCELLENT\nColor bar max of {viz_max} captures > 99.5% of data\nNo significant outliers"
        color = "lightgreen"
    elif retention_rate >= 95:
        assessment = f"✓ GOOD\nColor bar max of {viz_max} captures > 95% of data,\nFiltering some outliers"
        color = "lightblue"
    elif retention_rate >= 90:
        assessment = "⚠ ACCEPTABLE\nSome data filtered,\nconsider increasing viz_max"
        color = "lightyellow"
    else:
        assessment = (
            "⚠ REVIEW NEEDED\nSignificant data loss,\nincrease viz_max recommended"
        )
        color = "lightcoral"

    ax5.text(
        0.5,
        0.5,
        assessment,
        transform=ax5.transAxes,
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.4", facecolor=color, alpha=0.9),
    )
    ax5.set_title("Color Bar Max Value Assessment", fontsize=14, fontweight="bold")

    plt.tight_layout()

    # Save with high quality
    if output_path is not None:
        filename = f"{variable}_viz_max_justification.png"
        plt.savefig(
            Path(output_path, filename),
            # bbox_inches="tight",
            dpi=300,
            facecolor="white",
            edgecolor="none",
        )
        print(f"Enhanced viz max justification plot saved as: {filename}")
    else:
        plt.show()

    plt.close()


def create_standard_comparison_plots(
    region_stats, variable, variable_display_name, units, regions, output_path
):
    """Create the standard comparison plots (simplified version of original)"""

    # Create comparison table
    table_data = []
    metrics = set()
    region_colors = {region: plt.cm.tab10(i) for i, region in enumerate(regions)}

    for stat in region_stats:
        if stat["region"] is not None:
            row = {"Region": stat["region"]}
            for metric, value in stat["stats"].items():
                row[metric] = value
                metrics.add(metric)
            table_data.append(row)

    comparison_table = pd.DataFrame(table_data)

    # Simple regional KDE comparison
    plt.figure(figsize=(12, 6))

    for stat in region_stats:
        if stat["region"] is not None and "df" in stat:
            region = stat["region"]
            sns.kdeplot(stat["df"][variable], label=region, color=region_colors[region])

    plt.title(f"Regional Comparison: {variable_display_name}")
    plt.xlabel(f"{variable_display_name} ({units})")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()

    if output_path is not None:
        filename = f"{variable}_regional_comparison.png"
        plt.savefig(Path(output_path, filename), bbox_inches="tight", dpi=300)
        print(f"Regional comparison plot saved as: {filename}")
    else:
        plt.show()

    plt.close()


def standardize_units(variable: str, current_units: str) -> str:
    """Standardize units based on variable type"""
    variable_lower = variable.lower()

    if "speed" in variable_lower or "velocity" in variable_lower:
        return "m/s"
    elif "power_density" in variable_lower or "power density" in variable_lower:
        return "W/m²"
    elif any(
        term in variable_lower for term in ["distance", "depth", "height", "length"]
    ):
        return "m"
    else:
        return current_units


def compile_results(
    variable, variable_display_name, units, regions, region_stats, all_data, viz_max
):
    """Compile and return analysis results"""

    # Create comparison table
    table_data = []
    for stat in region_stats:
        if stat["region"] is not None:
            row = {"Region": stat["region"]}
            for metric, value in stat["stats"].items():
                row[metric] = value
            table_data.append(row)

    comparison_table = pd.DataFrame(table_data)
    region_colors = {region: plt.cm.tab10(i) for i, region in enumerate(regions)}

    result = {
        "variable": variable,
        "variable_display_name": variable_display_name,
        "regions": regions,
        "comparison_table": comparison_table,
        "region_colors": region_colors,
        "units": units,
    }

    # Add validation metrics if viz_max provided
    if viz_max is not None and len(all_data) > 0:
        retained_points = np.sum(all_data <= viz_max)
        total_points = len(all_data)

        result["validation"] = {
            "viz_max": viz_max,
            "total_points": total_points,
            "retained_points": retained_points,
            "retention_rate": (retained_points / total_points) * 100,
            "global_p99": np.percentile(all_data, 99),
            "global_p999": np.percentile(all_data, 99.9),
            "global_max": np.max(all_data),
        }

        # Print summary
        print("\nViz Max Justification Summary:")
        print(f"Variable: {variable_display_name} ({units})")
        print(f"Chosen viz_max: {viz_max} {units}")
        print(
            f"Data retention: {result['validation']['retention_rate']:.1f}% ({retained_points:,}/{total_points:,} points)"
        )
        print(f"99th percentile: {result['validation']['global_p99']:.1f} {units}")

    return result


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

        # Process regional images (these remain the same as they're from individual region analysis)
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

        # Process NEW comparison images from base directory - updated for new naming scheme
        comparison_files = [
            # Viz max justification plots (primary new outputs)
            "vap_water_column_mean_sea_water_speed_viz_max_justification.png",
            "vap_water_column_95th_percentile_sea_water_speed_viz_max_justification.png",
            "vap_water_column_mean_sea_water_power_density_viz_max_justification.png",
            "vap_water_column_95th_percentile_sea_water_power_density_viz_max_justification.png",
            "vap_sea_floor_depth_viz_max_justification.png",
            # Regional comparison plots (simplified KDE plots)
            "vap_water_column_mean_sea_water_speed_regional_comparison.png",
            "vap_water_column_95th_percentile_sea_water_speed_regional_comparison.png",
            "vap_water_column_mean_sea_water_power_density_regional_comparison.png",
            "vap_water_column_95th_percentile_sea_water_power_density_regional_comparison.png",
            "vap_sea_floor_depth_regional_comparison.png",
            # Bar comparison plots (if still generated by your updated function)
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

        # Copy NEW comparison images - updated file list
        comparison_files = [
            # Viz max justification plots
            "vap_water_column_mean_sea_water_speed_viz_max_justification.png",
            "vap_water_column_95th_percentile_sea_water_speed_viz_max_justification.png",
            "vap_water_column_mean_sea_water_power_density_viz_max_justification.png",
            "vap_water_column_95th_percentile_sea_water_power_density_viz_max_justification.png",
            "vap_sea_floor_depth_viz_max_justification.png",
            # Regional comparison plots
            "vap_water_column_mean_sea_water_speed_regional_comparison.png",
            "vap_water_column_95th_percentile_sea_water_speed_regional_comparison.png",
            "vap_water_column_mean_sea_water_power_density_regional_comparison.png",
            "vap_water_column_95th_percentile_sea_water_power_density_regional_comparison.png",
            "vap_sea_floor_depth_regional_comparison.png",
            # Bar comparison plots (if still generated)
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
                range_str = f"{min_val:.0f} - {max_val:.0f}"
            elif max_value >= 10:
                range_str = f"{min_val:.2f} - {max_val:.2f}"
            else:
                range_str = f"{min_val:.2f} - {max_val:.2f}"

            color_levels.append(
                {
                    "level": i + 1,
                    # "range": f"{range_str} [{units}]",
                    "range": f"{range_str} [{units}]",
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
            max_val = bounds[i] if i > 0 else bounds[0]
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
    parquet_paths,
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

    viz_specs = {
        "mean_sea_water_speed": {
            "title": "Mean Sea Water Speed",
            "units": "m/s",
            "column_name": "vap_water_column_mean_sea_water_speed",
            "colormap": "cmocean.thermal",
            "range_min": SEA_WATER_SPEED_CBAR_MIN,
            "range_max": SEA_WATER_SPEED_CBAR_MAX,
            "levels": SEA_WATER_SPEED_LEVELS,
            "physical_meaning": "Yearly average of depth averaged current speed",
            "intended_usage": "Site screening and turbine selection for power generation",
            "intended_usage_detail": "Primary metric for identifying viable tidal energy sites. Used to estimate annual energy production (AEP), compare site potential across regions, determine minimum viable current speeds for commercial deployment (typically >1.5 m/s), and select appropriate turbine technology. Critical for feasibility studies and initial resource assessments.",
            "equation": r"$\overline{\overline{U}} = U_{\text{average}} = \text{mean}\left(\left[\text{mean}(U_{1,t}, ..., U_{N_{\sigma},t}) \text{ for } t=1,...,T\right]\right)$",
            "equation_variables": "Where $U_{i,t} = \\sqrt{u_{i,t}^2 + v_{i,t}^2}$ are velocity magnitudes at uniformly distributed sigma level $i$ at volume centers at time $t$ (m/s), $N_{\\sigma} = 10$ levels, $T = 1$ year",
        },
        "p95_sea_water_speed": {
            "title": "95th Percentile Sea Water Speed",
            "units": "m/s",
            "column_name": "vap_water_column_95th_percentile_sea_water_speed",
            "colormap": "cmocean.matter",
            "range_min": SEA_WATER_SPEED_CBAR_MIN,
            "range_max": SEA_WATER_MAX_SPEED_CBAR_MAX,
            "levels": SEA_WATER_MAX_SPEED_LEVELS,
            "physical_meaning": "95th percentile of yearly depth maximum current speed",
            "intended_usage": "Generator sizing and power electronics design",
            "intended_usage_detail": "Critical for sizing electrical generation components. Used to determine maximum generator output capacity, size power electronics and converters for peak electrical loads, design control systems for extreme speed conditions, and set cut-out speeds for generator protection. Essential for electrical system certification, grid connection requirements, and ensuring generators can handle maximum rotational speeds without damage.",
            "equation": r"$U_{95} = \text{percentile}_{95}\left(\left[\max(U_{1,t}, ..., U_{N_{\sigma},t}) \text{ for } t=1,...,T\right]\right)$",
            "equation_variables": "Where $U_{i,t} = \\sqrt{u_{i,t}^2 + v_{i,t}^2}$ are velocity magnitudes at uniformly distributed sigma level $i$ at volume centers at time $t$ (m/s), $N_{\\sigma} = 10$ levels, $T = 1$ year",
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
            "intended_usage": "Resource quantification and economic feasibility analysis",
            "intended_usage_detail": "Direct measure of extractable energy resource for economic analysis. Used to calculate theoretical power output, estimate capacity factors for project financing, compare energy density between sites, and determine optimal turbine spacing in arrays. Essential for LCOE calculations, investor presentations, and grid integration planning. Minimum thresholds (typically >300 W/m²) define commercial viability.",
            "equation": r"$\overline{\overline{P}} = P_{\text{average}} = \text{mean}\left(\left[\text{mean}(P_{1,t}, ..., P_{N_{\sigma},t}) \text{ for } t=1,...,T\right]\right)$",
            "equation_variables": "Where $P_{i,t} = \\frac{1}{2} \\rho U_{i,t}^3$ with $\\rho = 1025$ kg/m³, $U_{i,t}$ are velocity magnitudes at uniformly distributed sigma level $i$ at volume centers at time $t$, $N_{\\sigma} = 10$ levels, $T = 1$ year",
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
            "intended_usage": "Structural design loads and extreme loading conditions",
            "intended_usage_detail": "Essential for structural engineering and extreme load analysis. Used to determine maximum design loads for turbine blades, drive trains, support structures, and foundation systems. Critical for fatigue analysis, ultimate load calculations, and ensuring structural integrity during extreme tidal events. Defines design margins for mooring systems, tower structures, and emergency braking systems. Required for structural certification and insurance assessments.",
            "equation": r"$P_{95} = \text{percentile}_{95}\left(\left[\max(P_{1,t}, ..., P_{N_{\sigma},t}) \text{ for } t=1,...,T\right]\right)$",
            "equation_variables": "Where $P_{i,t} = \\frac{1}{2} \\rho U_{i,t}^3$ with $\\rho = 1025$ kg/m³, $U_{i,t}$ are velocity magnitudes at uniformly distributed sigma level $i$ at volume centers at time $t$, $N_{\\sigma} = 10$ levels, $T = 1$ year",
        },
        "distance_to_sea_floor": {
            "title": "Mean Depth",
            "units": "m (below NAVD88)",
            "column_name": "vap_sea_floor_depth",
            "colormap": "cmocean.deep",
            "range_min": SEA_FLOOR_DEPTH_MIN,
            "range_max": SEA_FLOOR_DEPTH_MAX,
            "levels": SEA_FLOOR_DEPTH_LEVELS,
            "physical_meaning": "Yearly average distance from water surface to the sea floor",
            "intended_usage": "Installation planning and foundation design",
            "intended_usage_detail": "Fundamental constraint for deployment strategy and cost estimation. Used to determine installation vessel requirements, foundation type selection (gravity, pile, suction caisson), and deployment method feasibility. Critical for cost modeling (deeper = more expensive), accessibility planning for maintenance operations, and environmental impact assessments. Optimal depths typically 20-50m for current technology, with deeper sites requiring specialized equipment and higher costs.",
            "equation": r"$\overline{d} = d_{\text{average}} = \text{mean}\left(\left[(h + \zeta_t) \text{ for } t=1,...,T\right]\right)$",
            "equation_variables": "Where $h$ is bathymetry below NAVD88 (m), $\\zeta_t$ is sea surface elevation above NAVD88 at time $t$ (m), $T = 1$ year",
        },
    }

    # Markdown content
    md_content = []

    # Header
    md_content.extend(
        [
            "# ME Atlas High Resolution Tidal Data QOI Visualization Specification",
            "",
            "The following sections provide the specification for visualizing selected high resolution tidal hindcast variables on the [NREL Marine Energy Atlas](https://maps.nrel.gov/marine-energy-atlas/data-viewer/data-library/layers?vL=WavePowerMerged)",
            "",
        ]
    )
    # Add location filepath details
    md_content.extend(
        [
            "",
            "## Available Data File Details",
            "",
            "Base directory for all data files:",
            "",
            f"* <base_dir>: `{config['dir']['base']}`",
            "",
            "| Location Name | System | File Path |",
            "| --- | --- | --- |",
        ]
    )

    for this_region, parquet_path in parquet_paths.items():
        loc_spec = config["location_specification"]
        region_name = None
        for loc in loc_spec.values():
            if loc["output_name"] == this_region:
                region_name = loc["label"]

        md_content.append(
            f"| {region_name} | NREL Kestrel HPC | `{str(parquet_path).replace(config['dir']['base'], '<base_dir>')}` |"
        )

    # Add Location Details
    md_content.extend(
        [
            "",
            "## Location Details",
            "",
            "| Location Name | Face Count | Averaging Dates [UTC] | Averaging Temporal Resolution",
            "| --- | --- | --- | --- |",
        ]
    )

    for this_region in parquet_paths.keys():
        loc_spec = config["location_specification"]
        this_loc = None
        for loc in loc_spec.values():
            if loc["output_name"] == this_region:
                this_loc = loc

        md_content.append(
            f"| {this_loc['label']} | {this_loc['face_count']:,} | {this_loc['start_date_utc']} to {this_loc['end_date_utc']} | {this_loc['temporal_resolution']} |"
        )

    md_content.extend(
        [
            "",
            "## Variable Overview",
            "",
            "| Variable | Units | Data Column |",
            "| -------- | ----- | ----------- |",
        ]
    )

    for var in viz_specs.values():
        md_content.append(f"| {var['title']} | {var['units']} | {var['column_name']} |")

    md_content.extend(
        [
            "",
            "## Variable Usage",
            "",
            "| Variable | Meaning | Intended Usage",
            "| ---- | ------- | --- |",
        ]
    )

    for var in viz_specs.values():
        md_content.append(
            f"| {var['title']} | {var['physical_meaning']} | {var['intended_usage']} |"
        )

    md_content.extend(
        [
            "",
            "## Variable Equations",
            "",
        ]
    )

    for var in viz_specs.values():
        md_content.append(
            f"### {var['title']}" "",
            "",
            "Equation:",
            "",
            f"{var['equation']}" f"",
            f"{var['equation_variables']} |",
        )

    # Add coordinate details
    md_content.extend(
        [
            "",
            "## Coordinate Details",
            "",
            "The high resolution tidal hindcast data is based on an unstructured three dimensional grid of triangular faces with variable resolution.",
            "To visualize in two dimensions (lat/lon) the data for all depths are combined (averaging, or 95th percentile of maximums) into a single layer.",
            "This single layer has coordinates defined at the center and corners of each triangular element.",
            "Within the parquet files the coordinates are stored in the following columns:",
            "",
            "Notes:" "",
            "* All coordinates are in WGS84 (EPSG:4326) format.",
            "* All centerpoints have been validated to be within the bounding box of the triangular element.",
            "* All triangular elements coordinates are visualized below and can be assumed to be valid",
            "* Triangular elements are not guaranteed to be equilateral or isosceles, and may have varying angles and lengths.",
            "* Triangular elements vertice order has not been validated to be consistent across all regions.",
            "* The Aleutian Islands, Alaska dataset has elements that cross the from -180 to 180 longitude, which may cause visual artifacts in some mapping software.",
            "",
            "| Column Name | Description",
            "| --- | --- |",
            "| `lat_center` | Element Center Latitude",
            "| `lon_center` | Element Center Longitude",
            "| `element_corner_1_lat` | Element Triangular Vertice 1 Latitude",
            "| `element_corner_1_lon` | Element Triangular Vertice 1 Longitude",
            "| `element_corner_2_lat` | Element Triangular Vertice 2 Latitude",
            "| `element_corner_2_lon` | Element Triangular Vertice 2 Longitude",
            "| `element_corner_3_lat` | Element Triangular Vertice 3 Latitude",
            "| `element_corner_3_lon` | Element Triangular Vertice 3 Longitude",
        ]
    )

    # Add detailed specifications section
    md_content.extend(
        [
            "## Color Details",
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

    md_content.append("")

    # Add color mapping details if available
    if color_level_data:
        md_content.extend(
            [
                "## Color Specifications",
                "",
                "The following tables provide exact color specifications for each variable.",
                "All colors use discrete levels with an overflow level for values exceeding the maximum range.",
                "",
            ]
        )

        for var_key, spec in viz_specs.items():
            if var_key in color_level_data:
                md_content.extend(
                    [
                        f"### {spec['title']} [{spec['units']}], `{spec['column_name']}`",
                        "",
                        f"* **Colormap:** {spec['colormap']}",
                        f"* **Data Range:** {spec['range_min']} to {spec['range_max']} {spec['units']}",
                        f"* **Discrete Levels:** {spec['levels'] + 1} ({spec['levels']} within range + 1 overflow level)",
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
                    color_preview = f'<span style="background-color:{hex_color}; color:{hex_color}; padding:2px 8px; border-radius:3px;">████</span>'
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
            "## Visualizations by Variable",
            "",
        ]
    )
    # Add each visualization with image embed and caption
    viz_types = [
        ("mean_sea_water_speed", "Mean Sea Water Speed"),
        ("p95_sea_water_speed", "95th Percentile Sea Water Speed"),
        ("mean_sea_water_power_density", "Mean Sea Water Power Density"),
        ("p95_sea_water_power_density", "95th Percentile Sea Water Power Density"),
        ("distance_to_sea_floor", "Distance to Sea Floor"),
    ]
    for viz_key, viz_title in viz_types:
        md_content.extend(
            [
                f"### {viz_title}",
                "",
            ]
        )
        for region in regions_processed:
            # region_title = region.replace("_", " ").title()
            loc_spec = config["location_specification"]
            this_loc = None
            for loc in loc_spec.values():
                if loc["output_name"] == region:
                    this_loc = loc
            region_title = this_loc["label"]
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
                    f"**{region_title} {viz_title}**",
                    "",
                    f"![{viz_title} for {region_title}]({img_path})",
                    f"*Figure: {viz_title} spatial distribution for {region_title}. Units: {units}",
                    "",
                ]
            )

        md_content.extend(["", "---", ""])

    # Cross-regional analysis section with images
    md_content.extend(
        [
            "## Cross-Regional Comparative Analysis",
            "",
            "Comparative visualizations across all processed regions provide insights into spatial variability, statistical patterns, and visualization parameter validation.",
            "",
            "### Visualization Maximum Justification",
            "",
            "These comprehensive plots validate the chosen visualization maximum (viz_max) parameters used throughout the analysis. Each visualization demonstrates that the selected cutoff values effectively capture the bulk of the data while filtering extreme outliers, ensuring meaningful and readable visualizations.",
            "",
        ]
    )

    # Viz max justification plots - these are the main new outputs
    viz_justification_plots = [
        (
            "vap_water_column_mean_sea_water_speed_viz_max_justification.png",
            "Mean Sea Water Speed",
            "m/s",
            "Validates the visualization maximum used for mean sea water speed analysis, showing data retention rates and outlier filtering effectiveness.",
        ),
        (
            "vap_water_column_95th_percentile_sea_water_speed_viz_max_justification.png",
            "95th Percentile Sea Water Speed",
            "m/s",
            "Demonstrates the appropriateness of the visualization cutoff for 95th percentile sea water speed values across all regions.",
        ),
        (
            "vap_water_column_mean_sea_water_power_density_viz_max_justification.png",
            "Mean Sea Water Power Density",
            "W/m²",
            "Justifies the power density visualization maximum by showing statistical distribution and outlier characteristics.",
        ),
        (
            "vap_water_column_95th_percentile_sea_water_power_density_viz_max_justification.png",
            "95th Percentile Sea Water Power Density",
            "W/m²",
            "Validates the visualization bounds for 95th percentile power density measurements across regional datasets.",
        ),
        (
            "vap_sea_floor_depth_viz_max_justification.png",
            "Sea Floor Depth",
            "m",
            "Shows the effectiveness of depth visualization parameters in capturing bathymetric variability while controlling for extreme outliers.",
        ),
    ]

    for img_file, title, units, description in viz_justification_plots:
        img_path = f"docs/img/{img_file}"
        md_content.extend(
            [
                f"**{title} - Visualization Maximum Validation**",
                "",
                f"![{title} Viz Max Justification]({img_path})",
                f"*Figure: Comprehensive validation of visualization maximum for {title.lower()}. Shows full data distribution, regional comparisons within bounds, key statistics, and outlier assessment. Units: {units}. {description}*",
                "",
            ]
        )

    md_content.extend(
        [
            "### Regional Distribution Comparisons",
            "",
            "These kernel density estimation (KDE) plots provide clean statistical comparisons of variable distributions across all processed regions, focused within the validated visualization ranges.",
            "",
        ]
    )

    # Regional comparison plots - these are the simplified KDE plots
    regional_comparison_plots = [
        (
            "vap_water_column_mean_sea_water_speed_regional_comparison.png",
            "Mean Sea Water Speed",
            "m/s",
            "Regional distribution patterns for mean sea water speed",
        ),
        (
            "vap_water_column_95th_percentile_sea_water_speed_regional_comparison.png",
            "95th Percentile Sea Water Speed",
            "m/s",
            "Comparative analysis of high-speed current characteristics across regions",
        ),
        (
            "vap_water_column_mean_sea_water_power_density_regional_comparison.png",
            "Mean Sea Water Power Density",
            "W/m²",
            "Power density distribution comparison highlighting regional resource potential",
        ),
        (
            "vap_water_column_95th_percentile_sea_water_power_density_regional_comparison.png",
            "95th Percentile Sea Water Power Density",
            "W/m²",
            "High-power density event comparison across different oceanic regions",
        ),
        (
            "vap_sea_floor_depth_regional_comparison.png",
            "Sea Floor Depth",
            "m",
            "Bathymetric distribution comparison showing depth characteristics by region",
        ),
    ]

    for img_file, title, units, description in regional_comparison_plots:
        img_path = f"docs/img/{img_file}"
        md_content.extend(
            [
                f"**{title} Distribution Comparison**",
                "",
                f"![{title} Regional Comparison]({img_path})",
                f"*Figure: Kernel density estimation comparison of {title.lower()} across all processed regions. Units: {units}. {description}. Distributions are shown within validated visualization bounds for optimal clarity.*",
                "",
            ]
        )

    # Note: The original bar chart comparisons may still exist if your analyze_variable_across_regions
    # function still generates them. If so, you can keep this section:
    md_content.extend(
        [
            "### Percentile Bar Chart Comparisons",
            "",
            "These charts provide quantitative comparison of key percentile values across regions, with visualization maximum reference lines for context.",
            "",
        ]
    )

    bar_comparisons = [
        (
            "vap_water_column_mean_sea_water_speed_bar_comparison.png",
            "Mean Sea Water Speed",
            "m/s",
            "Quantitative percentile comparison with visualization bounds overlay",
        ),
        (
            "vap_water_column_95th_percentile_sea_water_speed_bar_comparison.png",
            "95th Percentile Sea Water Speed",
            "m/s",
            "High-speed percentile values across regions with reference thresholds",
        ),
        (
            "vap_water_column_mean_sea_water_power_density_bar_comparison.png",
            "Mean Sea Water Power Density",
            "W/m²",
            "Power density percentile analysis with visualization maximum context",
        ),
        (
            "vap_water_column_95th_percentile_sea_water_power_density_bar_comparison.png",
            "95th Percentile Sea Water Power Density",
            "W/m²",
            "Regional power density extremes with validated cutoff references",
        ),
        (
            "vap_sea_floor_depth_bar_comparison.png",
            "Sea Floor Depth",
            "m",
            "Depth percentile comparison across bathymetric regions",
        ),
    ]

    for img_file, title, units, description in bar_comparisons:
        img_path = f"docs/img/{img_file}"
        md_content.extend(
            [
                f"**{title} Percentile Comparison**",
                "",
                f"![{title} Bar Comparison]({img_path})",
                f"*Figure: Percentile values of {title.lower()} compared across all processed regions. Units: {units}. {description}. Enables quantitative assessment of regional variability and extreme value characteristics.*",
                "",
            ]
        )

    # Add a summary section explaining the visualization approach
    md_content.extend(
        [
            "### Visualization Methodology Notes",
            "",
            "**Visualization Maximum (Viz Max) Approach**: All visualizations use validated maximum values that capture 95-99.9% of the data while filtering extreme outliers. This approach ensures:",
            "",
            "- Clear, readable visualizations without distortion from extreme values",
            "- Consistent scales across regional comparisons",
            "- Transparent documentation of data filtering decisions",
            "- Preservation of statistical integrity for the bulk of the dataset",
            "",
            "**Data Retention**: The justification plots show exactly what percentage of data is retained vs. filtered, providing full transparency about the visualization choices and their impact on the analysis.",
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
    # regions = regions[:4]
    print("Available regions:")
    for i, region in enumerate(regions):
        print(f"{i+1}. {region}")

    mean_speed_stats = []
    max_speed_stats = []
    mean_power_density_stats = []
    max_power_density_stats = []
    sea_floor_depth_stats = []

    color_level_data = {}

    parquet_paths = {}

    for this_region in regions:
        # Get the parquet file path
        parquet_file = get_parquet_path(this_region)
        print(f"Reading file: {parquet_file}")

        parquet_paths[this_region] = str(parquet_file)

        # Read the parquet file
        df = pd.read_parquet(parquet_file)

        this_output_path = Path(VIZ_OUTPUT_DIR, this_region)
        this_output_path.mkdir(parents=True, exist_ok=True)

        print(f"\tPlotting {this_region} mean_sea_water_speed...")

        mean_speed_stats.append(
            analyze_variable(
                df,
                "vap_water_column_mean_sea_water_speed",
                "Mean Sea Water Speed",
                this_region,
                output_path=Path(this_output_path),
            )
        )

        fig, ax, color_data = plot_tidal_variable(
            df,
            this_region,
            "vap_water_column_mean_sea_water_speed",
            "Mean Sea Water Speed",
            SEA_WATER_SPEED_UNITS,
            SEA_WATER_SPEED_CBAR_MIN,
            SEA_WATER_SPEED_CBAR_MAX,
            is_aleutian="aleutian" in this_region,
            cmap=MEAN_SPEED_CMAP,
            save_path=Path(
                this_output_path,
                f"{this_region}_mean_sea_water_speed.png",
            ),
            n_colors=SEA_WATER_SPEED_LEVELS,
        )

        if "mean_sea_water_speed" not in color_level_data:
            color_level_data["mean_sea_water_speed"] = color_data

        plt.close()
        print(f"\tPlotting {this_region} p95_sea_water_speed...")

        # Capture color level data for markdown generation

        max_speed_stats.append(
            analyze_variable(
                df,
                "vap_water_column_95th_percentile_sea_water_speed",
                "95th Percentile Sea Water Speed",
                this_region,
                output_path=Path(this_output_path),
            )
        )

        fig, ax, color_data = plot_tidal_variable(
            df,
            this_region,
            "vap_water_column_95th_percentile_sea_water_speed",
            "95th Percentile Sea Water Speed",
            SEA_WATER_SPEED_UNITS,
            SEA_WATER_SPEED_CBAR_MIN,
            SEA_WATER_MAX_SPEED_CBAR_MAX,
            is_aleutian="aleutian" in this_region,
            cmap=MAX_SPEED_CMAP,
            save_path=Path(
                this_output_path,
                f"{this_region}_p95_sea_water_speed.png",
            ),
            n_colors=SEA_WATER_MAX_SPEED_LEVELS,
        )

        if "p95_sea_water_speed" not in color_level_data:
            color_level_data["p95_sea_water_speed"] = color_data

        plt.close()

        print(f"\tPlotting {this_region} mean_sea_water_power_density...")

        mean_power_density_stats.append(
            analyze_variable(
                df,
                "vap_water_column_mean_sea_water_power_density",
                "Mean Sea Water Power Density",
                this_region,
                output_path=Path(this_output_path),
            )
        )

        fig, ax, color_data = plot_tidal_variable(
            df,
            this_region,
            "vap_water_column_mean_sea_water_power_density",
            "Mean Sea Water Power Density",
            SEA_WATER_POWER_DENSITY_UNITS,
            SEA_WATER_POWER_DENSITY_CBAR_MIN,
            SEA_WATER_POWER_DENSITY_CBAR_MAX,
            is_aleutian="aleutian" in this_region,
            cmap=MEAN_POWER_DENSITY_CMAP,
            save_path=Path(
                this_output_path,
                f"{this_region}_mean_sea_water_power_density.png",
            ),
            n_colors=SEA_WATER_POWER_DENSITY_LEVELS,
        )

        if "mean_sea_water_power_density" not in color_level_data:
            color_level_data["mean_sea_water_power_density"] = color_data

        plt.close()

        print(f"\tPlotting {this_region} p95_sea_water_power_density...")

        max_power_density_stats.append(
            analyze_variable(
                df,
                "vap_water_column_95th_percentile_sea_water_power_density",
                "95th Percentile Sea Water Power Density",
                this_region,
                output_path=Path(this_output_path),
            )
        )

        fig, ax, color_data = plot_tidal_variable(
            df,
            this_region,
            "vap_water_column_95th_percentile_sea_water_power_density",
            "Max Sea Water Power Density",
            SEA_WATER_POWER_DENSITY_UNITS,
            SEA_WATER_POWER_DENSITY_CBAR_MIN,
            SEA_WATER_MAX_POWER_DENSITY_CBAR_MAX,
            is_aleutian="aleutian" in this_region,
            cmap=MAX_POWER_DENSITY_CMAP,
            save_path=Path(
                this_output_path,
                f"{this_region}_p95_sea_water_power_density.png",
            ),
            n_colors=SEA_WATER_MAX_POWER_DENSITY_LEVELS,
        )

        if "p95_sea_water_power_density" not in color_level_data:
            color_level_data["p95_sea_water_power_density"] = color_data

        plt.close()

        sea_floor_depth_stats.append(
            analyze_variable(
                df,
                "vap_sea_floor_depth",
                "Sea Floor Depth",
                this_region,
                output_path=Path(this_output_path),
            )
        )
        fig, ax, color_data = plot_tidal_variable(
            df,
            this_region,
            "vap_sea_floor_depth",
            "Distance to Sea Floor",
            "m",
            SEA_FLOOR_DEPTH_MIN,
            SEA_FLOOR_DEPTH_MAX,
            is_aleutian="aleutian" in this_region,
            cmap=SEA_FLOOR_DEPTH_CMAP,
            save_path=Path(
                this_output_path,
                f"{this_region}_distance_to_sea_floor.png",
            ),
            n_colors=SEA_FLOOR_DEPTH_LEVELS,
        )

        if "sea_floor_depth" not in color_level_data:
            color_level_data["sea_floor_depth"] = color_data

        plt.close()
        sea_floor_depth_stats.append(
            analyze_variable(
                df,
                "vap_sea_floor_depth",
                "Sea Floor Depth",
                this_region,
                output_path=Path(this_output_path),
            )
        )
        fig, ax, color_data = plot_tidal_variable(
            df,
            this_region,
            "vap_sea_floor_depth",
            "Distance to Sea Floor",
            "m",
            SEA_FLOOR_DEPTH_MIN,
            SEA_FLOOR_DEPTH_MAX,
            is_aleutian="aleutian" in this_region,
            cmap=SEA_FLOOR_DEPTH_CMAP,
            save_path=Path(
                this_output_path,
                f"{this_region}_distance_to_sea_floor.png",
            ),
            n_colors=SEA_FLOOR_DEPTH_LEVELS,
        )

        if "sea_floor_depth" not in color_level_data:
            color_level_data["sea_floor_depth"] = color_data

        plt.close()

    sns.set_theme()

    print("Calculating and Plotting Speed variable_summary...")
    mean_speed_summary = analyze_variable_across_regions(
        mean_speed_stats, output_path=VIZ_OUTPUT_DIR, viz_max=SEA_WATER_SPEED_CBAR_MAX
    )
    max_speed_summary = analyze_variable_across_regions(
        max_speed_stats,
        output_path=VIZ_OUTPUT_DIR,
        viz_max=SEA_WATER_MAX_SPEED_CBAR_MAX,
    )
    mean_power_density_summary = analyze_variable_across_regions(
        mean_power_density_stats,
        output_path=VIZ_OUTPUT_DIR,
        viz_max=SEA_WATER_POWER_DENSITY_CBAR_MAX,
    )
    max_power_density_summary = analyze_variable_across_regions(
        max_power_density_stats,
        output_path=VIZ_OUTPUT_DIR,
        viz_max=SEA_WATER_MAX_POWER_DENSITY_CBAR_MAX,
    )
    sea_floor_depth_summary = analyze_variable_across_regions(
        sea_floor_depth_stats, output_path=VIZ_OUTPUT_DIR, viz_max=SEA_FLOOR_DEPTH_MAX
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
        parquet_paths=parquet_paths,
    )

    print("Analysis and documentation complete!")
