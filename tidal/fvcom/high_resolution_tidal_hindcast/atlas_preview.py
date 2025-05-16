from pathlib import Path
from typing import Any, Dict, List

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
SEA_WATER_MAX_SPEED_CBAR_MAX = 5.0

SEA_WATER_POWER_DENSITY_CBAR_MAX = 4000  # 0.5 * 1025 * (2.0^3) = 4,100
SEA_WATER_POWER_DENSITY_CBAR_MIN = 0
SEA_WATER_MAX_POWER_DENSITY_CBAR_MAX = 64000  # 0.5 * 1025 * (5.0^3) = 64,062.5
SEA_FLOOR_DEPTH_MIN = 0
SEA_FLOOR_DEPTH_MAX = 250
BASEMAP_PROVIDER = ctx.providers.Esri.WorldImagery

SEA_WATER_SPEED_UNITS = r"$m/s$"
SEA_WATER_POWER_DENSITY_UNITS = r"$W/m^2$"

COLOR_BAR_DISCRETE_LEVELS = 8

MEAN_SPEED_CMAP = cmocean.cm.thermal
MAX_SPEED_CMAP = cmocean.cm.matter
MEAN_POWER_DENSITY_CMAP = cmocean.cm.dense
MAX_POWER_DENSITY_CMAP = cmocean.cm.tempo
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
        # Create discrete colormap
        base_cmap = plt.get_cmap(cmap)
        colors = base_cmap(np.linspace(0, 1, n_colors))
        discrete_cmap = mpl.colors.LinearSegmentedColormap.from_list(
            "discrete_cmap", colors, N=n_colors
        )

        # Create boundaries for the discrete levels
        bounds = np.linspace(vmin, vmax, n_colors + 1)
        discrete_norm = mpl.colors.BoundaryNorm(bounds, n_colors)
        cmap = discrete_cmap
    else:
        bounds = None

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

    _add_colorbar_and_title(
        fig,
        ax,
        scatter,
        label,
        units,
        title,
        location,
        discrete_levels=bounds if n_colors is not None else None,
    )

    plt.tight_layout()
    if show is True:
        plt.show()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    # Print out the color and value range for each level if discrete levels are used
    if n_colors is not None:
        _print_color_level_ranges(bounds, label, units, cmap, n_colors)

    return fig, ax


def _print_color_level_ranges(bounds, label, units, cmap, n_colors):
    """
    Print out the color and value range for each discrete color level.

    Parameters:
    -----------
    bounds : numpy array
        Array of boundary values for each color level
    label : str
        Label for the variable
    units : str
        Units for the variable
    cmap : matplotlib colormap
        Colormap being used
    n_colors : int
        Number of color levels
    """

    print(f"\n{label} Color Level Ranges [{units}]:")
    print("-" * 50)

    # Get the RGB values for each color level
    colors = [cmap(i / (n_colors - 1)) for i in range(n_colors)]

    # Format the RGB values as hex colors
    hex_colors = [mcolors.rgb2hex(color[:3]) for color in colors]

    # Print the range and color for each level
    for i in range(n_colors):
        if i < n_colors - 1:
            min_val = bounds[i]
            max_val = bounds[i + 1]
            level_str = f"Level {i+1}: {min_val:.4f} to {max_val:.4f} {units}"
            print(f"{level_str:<40} | Color: {hex_colors[i]}")
        else:
            # This should not happen with the way bounds are created, but just in case
            print(f"Level {i+1}: {bounds[i]:.4f} {units} | Color: {hex_colors[i]}")

    print("-" * 50)


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

        # Calculate midpoints between levels for tick positions
        midpoints = [
            (discrete_levels[i] + discrete_levels[i + 1]) / 2
            for i in range(len(discrete_levels) - 1)
        ]

        # Create labels showing range intervals
        tick_labels = []
        for i in range(len(discrete_levels) - 1):
            if i == len(discrete_levels) - 2:  # Last interval
                # For the last interval, use "≥" to indicate open-ended range
                tick_labels.append(f"[≥{tick_format % discrete_levels[i]})")
            else:
                tick_labels.append(
                    f"[{tick_format % discrete_levels[i]}-{tick_format % discrete_levels[i+1]})"
                )

        # Create the colorbar with specific ticks at midpoints
        cbar = fig.colorbar(
            scatter,
            ax=ax,
            orientation="vertical",
            pad=0.02,
            fraction=0.03,
            shrink=0.7,
            ticks=midpoints,  # Use midpoints for tick positions
        )

        # Set custom tick labels showing the ranges
        cbar.ax.set_yticklabels(tick_labels)
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


def analyze_speed(
    df: pd.DataFrame, region_name: str, output_path: None
) -> Dict[str, Any]:
    """
    Analyze sea water speed statistics from sigma-layer data for a given region and plot
    histograms with KDE and percentile lines.

    Context: The speed data represents values calculated across 10 sigma layers from the
    original data. All measurements are in meters per second (m/s).

    Args:
        df: DataFrame containing the sea water speed data across sigma layers
        region_name: Name of the region being analyzed

    Returns:
        Dictionary containing region name, statistics, and data sample for KDE analysis
    """
    # Define columns to analyze - these are calculated across 10 sigma layers
    columns = [
        "vap_water_column_mean_sea_water_speed",
        "vap_water_column_95th_percentile_sea_water_speed",
        # "vap_water_column_max_sea_water_speed",
    ]

    # Create a figure with subplots - one for each column
    fig, axes = plt.subplots(len(columns), 1, figsize=(16, 5 * len(columns)))

    # Dictionary to store the results
    results = {
        "region": region_name,
        "df": df,  # Store the dataframe for use in the meta-analysis
        "stats": {},
    }

    # Calculate percentiles and plot for each column
    for i, col in enumerate(columns):
        ax = axes[i] if len(columns) > 1 else axes

        # Calculate key percentiles
        p95 = df[col].quantile(0.95)
        p99 = df[col].quantile(0.99)
        p9999 = df[col].quantile(0.9999)

        # Store only the statistics we need for comparison
        results["stats"][col] = {
            "p95": p95,
            "p99": p99,
            "p9999": p9999,
            "min": df[col].min(),
            "max": df[col].max(),
        }

        # Plot histogram with KDE
        sns.histplot(df[col], kde=True, ax=ax, bins=50, alpha=0.6)

        # Add vertical lines for percentiles
        ax.axvline(p95, color="r", linestyle="--")
        ax.axvline(p99, color="g", linestyle="--")
        ax.axvline(p9999, color="b", linestyle="--")

        # Add annotations
        ax.annotate(
            f"95%: {p95:.3f}",
            xy=(p95, 0),
            xytext=(p95, ax.get_ylim()[1] * 0.9),
            arrowprops=dict(arrowstyle="->"),
            ha="right",
        )
        ax.annotate(
            f"99%: {p99:.3f}",
            xy=(p99, 0),
            xytext=(p99, ax.get_ylim()[1] * 0.8),
            arrowprops=dict(arrowstyle="->"),
            ha="right",
        )
        ax.annotate(
            f"99.99%: {p9999:.3f}",
            xy=(p9999, 0),
            xytext=(p9999, ax.get_ylim()[1] * 0.7),
            arrowprops=dict(arrowstyle="->"),
            ha="left",
        )

        # Get display name from dictionary
        display_name = COLUMN_DISPLAY_NAMES.get(col, col)

        # Set title and labels
        ax.set_title(f"{region_name} - {display_name} (m/s)")
        ax.set_xlabel("Speed (m/s)")
        ax.set_ylabel("Frequency")

    # Additional combined histogram plot (all columns together)
    plt.figure(figsize=(16, 8))

    # Create the histogram with proper labels
    ax = df[columns].plot(
        kind="hist",
        title=f"{region_name} - Sea Water Speed Comparison",
        figsize=(16, 8),
        alpha=0.4,
        bins=50,
    )

    # Update the legend with more descriptive names
    labels = [COLUMN_DISPLAY_NAMES[col] for col in columns]
    ax.legend(labels)

    plt.xlabel("Speed (m/s)")
    plt.ylabel("Frequency")

    # Tight layout and show the individual plots
    fig.tight_layout()
    if output_path is not None:
        plt.savefig(Path(output_path, "speed_analysis.png"))
    else:
        plt.show()

    return results


def analyze_all_region_speed_statistics(
    region_stats: List[Dict[str, Any]], output_path=None
) -> Dict[str, Any]:
    """
    Perform meta-analysis comparing speed statistics across different regions.

    Args:
        region_stats: List of dictionaries containing region statistics and dataframes
                     from analyze_speed

    Returns:
        Dictionary with summary statistics and comparison tables
    """
    if not region_stats:
        print("No region statistics provided for analysis.")
        return {}

    # Extract regions
    regions = [stat["region"] for stat in region_stats]

    # Define columns to analyze
    columns = [
        "vap_water_column_mean_sea_water_speed",
        "vap_water_column_95th_percentile_sea_water_speed",
        "vap_water_column_max_sea_water_speed",
    ]

    # Create consistent color mapping for regions
    # Create a colormap with distinct colors for each region
    cmap = plt.get_cmap("tab10", len(regions))
    region_colors = {region: cmap(i) for i, region in enumerate(regions)}

    # Create comparison tables
    tables = {}
    for col in columns:
        # Create a list of dictionaries for this column
        col_data = []
        for stat in region_stats:
            if col in stat["stats"]:
                row = {
                    "Region": stat["region"],
                    "95%": stat["stats"][col]["p95"],
                    "99%": stat["stats"][col]["p99"],
                    "99.99%": stat["stats"][col]["p9999"],
                    "Min": stat["stats"][col]["min"],
                    "Max": stat["stats"][col]["max"],
                }
                col_data.append(row)

        # Create DataFrame
        tables[col] = pd.DataFrame(col_data)

    # Plot KDE comparisons
    for i, col in enumerate(columns):
        plt.figure(figsize=(16, 8))

        # Plot KDE for each region
        for stat in region_stats:
            if "df" in stat and col in stat["df"].columns:
                region = stat["region"]
                sns.kdeplot(stat["df"][col], label=region, color=region_colors[region])

        # Get display name from dictionary
        display_name = COLUMN_DISPLAY_NAMES.get(col, col)

        # Set title and labels
        plt.title(f"Comparison of {display_name} Across Regions")
        plt.xlabel("Speed (m/s)")
        plt.ylabel("Density")
        plt.legend()
        plt.tight_layout()
        if output_path is not None:
            plt.savefig(Path(output_path, f"{col}_kde_comparison.png"))
        else:
            plt.show()

    # Create bar charts for each percentile metric (95%, 99%, 99.99%)
    percentile_metrics = ["95%", "99%", "99.99%"]

    for col in columns:
        display_name = COLUMN_DISPLAY_NAMES.get(col, col)

        # Create a figure with subplots for each percentile
        fig, axes = plt.subplots(
            len(percentile_metrics), 1, figsize=(16, 5 * len(percentile_metrics))
        )
        fig.suptitle(
            f"{display_name} - Percentile Comparison Across Regions (m/s)", fontsize=16
        )

        # Get the data for this column
        df_col = tables[col]

        for i, metric in enumerate(percentile_metrics):
            # Sort data by the current percentile value
            sorted_data = df_col.sort_values(by=metric)

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
            ax.set_title(f"{metric} Values", fontsize=14)
            ax.set_xlabel("Region")
            ax.set_ylabel("Speed (m/s)")

            # Rotate x-tick labels for better readability
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for the suptitle
        if output_path is not None:
            plt.savefig(Path(output_path, f"{col}_bar_comparison.png"))
        else:
            plt.show()

    # Print comparison tables
    for col, table in tables.items():
        # Get display name from dictionary
        display_name = COLUMN_DISPLAY_NAMES.get(col, col)

        print(f"\n{display_name} Comparison (m/s):")
        print(table.to_string(index=False))

    # Return summary
    return {
        "regions": regions,
        "comparison_tables": tables,
        "region_colors": region_colors,  # Include color mapping for reference
    }


if __name__ == "__main__":
    # Display available regions
    regions = get_available_regions()
    # regions.reverse()
    print("Available regions:")
    for i, region in enumerate(regions):
        print(f"{i+1}. {region}")

    speed_loc_stats = []

    for i in range(0, 5):
        # Get the parquet file path
        selected_region = regions[i]
        parquet_file = get_parquet_path(selected_region)
        print(f"Reading file: {parquet_file}")

        # Read the parquet file
        df = pd.read_parquet(parquet_file)

        this_output_path = Path(VIZ_OUTPUT_DIR, selected_region)
        this_output_path.mkdir(parents=True, exist_ok=True)

        speed_loc_stats.append(
            analyze_speed(df, selected_region, output_path=Path(this_output_path))
        )

        print(f"\tPlotting {selected_region} mean_sea_water_speed...")

        plot_tidal_variable(
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
        )
        plt.clf()
        print(f"\tPlotting {selected_region} p95_sea_water_speed...")
        plot_tidal_variable(
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
        )
        plt.clf()

        print(f"\tPlotting {selected_region} mean_sea_water_power_density...")
        plot_tidal_variable(
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
        )
        plt.clf()

        print(f"\tPlotting {selected_region} p95_sea_water_power_density...")

        plot_tidal_variable(
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
        )

        plt.clf()
        if "vap_sea_floor_depth" in df.columns:
            plot_tidal_variable(
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
            )
            plt.clf()

    print("Calculating and Plotting Speed variable_summary...")
    speed_summary = analyze_all_region_speed_statistics(
        speed_loc_stats, output_path=VIZ_OUTPUT_DIR
    )
