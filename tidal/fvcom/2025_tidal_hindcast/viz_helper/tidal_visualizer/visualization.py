# visualization module - Part of tidal_visualizer package

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.colors import Normalize
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import contextily as ctx

from .core import TidalVisualizerBase
from .config import get_location_info, get_area_of_interest_info
from .io import find_dataset_file, load_dataset, get_variable_data
from .utils import (
    format_location_display,
    format_area_of_interest_display,
    format_combined_display,
)


class TidalVisualizer:
    def __init__(self, visualization_config, base_data_dir=None, disable_ssl=True):
        # Initialize the base class for core functionality
        self.base = TidalVisualizerBase(visualization_config, base_data_dir)
        self.config = visualization_config

        # Visualization settings
        self.figsize = (28, 14)
        self.dpi = 300
        self.alpha = 0.9
        self.font_scale = 1.75
        self.colormap = "viridis"
        self.basemap_provider = ctx.providers.Esri.WorldImagery
        # self.basemap_provider = ctx.providers.OpenStreetMap.Mapnik
        # self.use_tex = False
        self.use_tex = True
        self.output_formats = ["png"]  # Default to just png
        self.grid_line_width = 0.05  # Grid line width
        # self.grid_line_alpha = 1.0  # Grid line alpha/transparency
        self.grid_line_alpha = 0.0  # Grid line alpha/transparency

        # if disable_ssl is True:
        # configure_macos_corporate_ssl()
        # disable_ssl_for_contextily()
        #     disable_ssl_verification()

    def setup_fonts(self):
        import matplotlib.font_manager as fm

        # Add the font directory
        font_dir = "/home/asimms/.fonts"
        for font in fm.findSystemFonts(font_dir):
            fm.fontManager.addfont(font)

        # else:
        # Use Computer Modern Unicode font directly
        plt.rcParams.update(
            {
                "text.usetex": False,
                "font.family": "serif",
                "font.serif": [
                    "CMU Serif",
                    "cmunrm",
                ],  # CMU Serif is the font name, cmunrm is the file
                "font.size": 12 * self.font_scale,
            }
        )

    def setup_visualization_environment(self):
        """Set up the visualization environment."""
        plt.rcParams["figure.facecolor"] = "white"
        plt.rcParams["axes.facecolor"] = "white"
        plt.rcParams["savefig.facecolor"] = "white"
        plt.rcParams["figure.autolayout"] = True

    def handle_date_line_crossing(self, ds):
        """
        Handle datasets that cross the international date line.

        Parameters:
        -----------
        ds : xarray.Dataset
            The dataset containing lat_node and lon_node variables

        Returns:
        --------
        xarray.Dataset
            Dataset with normalized longitude values if date line crossing was detected,
            otherwise the original dataset
        bool
            Whether date line crossing was detected and corrected
        """

        # Extract longitude values
        lon_nodes = ds["lon_node"].values

        # Check if we have values on both sides of the date line
        has_negative = np.any(lon_nodes < 0)
        has_positive = np.any(lon_nodes > 0)

        # Calculate the span in original coordinates
        lon_min, lon_max = lon_nodes.min(), lon_nodes.max()
        original_span = lon_max - lon_min

        # Check if the data potentially crosses the date line
        # This is a heuristic - if we have both positive and negative values
        # and the span is large (more than 180 degrees)
        crosses_date_line = has_negative and has_positive and original_span > 180

        if crosses_date_line:
            print(
                f"Date line crossing detected! Original longitude span: {original_span:.2f}°"
            )

            # Create a copy of the dataset to modify
            ds_normalized = ds.copy(deep=True)

            # Convert all longitudes to [0, 360] range
            normalized_lons = np.where(lon_nodes < 0, lon_nodes + 360, lon_nodes)

            # Check if normalization reduces the span
            normalized_span = normalized_lons.max() - normalized_lons.min()

            if normalized_span < original_span:
                print(f"Normalized longitude span: {normalized_span:.2f}°")

                # Update the dataset with normalized longitudes
                ds_normalized["lon_node"].values = normalized_lons

                # Also update any other longitude variables if present
                if "lon" in ds:
                    ds_normalized["lon"].values = np.where(
                        ds["lon"].values < 0, ds["lon"].values + 360, ds["lon"].values
                    )

                return ds_normalized, True
            else:
                print(
                    "Normalization did not improve the span, using original coordinates"
                )

        # Return original if no crossing or normalization doesn't help
        return ds, False

    def create_mesh_triangulation_with_date_line_handling(
        self, ds, nodes_to_web_mercator=True
    ):
        """
        Create a triangulation for mesh visualization with date line handling.

        Parameters:
        -----------
        ds : xarray.Dataset
            The dataset containing lat_node and lon_node variables
        nodes_to_web_mercator : bool, optional
            If True, convert coordinates to Web Mercator projection

        Returns:
        --------
        matplotlib.tri.Triangulation
            Triangulation object for plotting
        tuple
            (x, y) coordinates of nodes
        bool
            Whether date line crossing was detected and handled
        """
        import matplotlib.tri as tri

        # Check and handle date line crossing
        ds_normalized, date_line_crossed = self.handle_date_line_crossing(ds)

        # Get coordinates from the (possibly normalized) dataset
        lat_nodes = ds_normalized["lat_node"].values
        lon_nodes = ds_normalized["lon_node"].values
        nv = ds_normalized["nv"].values - 1  # Adjust for 0-based indexing

        # Create the triangulation
        if nodes_to_web_mercator:
            # If coordinates were normalized, we need to carefully convert to Web Mercator
            # because the standard conversion might not work correctly for longitudes > 180
            if date_line_crossed:
                # Convert normalized longitudes back to -180 to 180 range for transformation
                # This is necessary because the projection typically expects this range
                lon_for_transform = np.where(
                    lon_nodes > 180, lon_nodes - 360, lon_nodes
                )
                x_web, y_web = self.base.transformer_to_web_mercator.transform(
                    lon_for_transform, lat_nodes
                )
            else:
                # Standard conversion for non-normalized coordinates
                x_web, y_web = self.base.transformer_to_web_mercator.transform(
                    lon_nodes, lat_nodes
                )
            return (
                tri.Triangulation(x_web, y_web, triangles=nv.T),
                (x_web, y_web),
                date_line_crossed,
            )
        else:
            # Use geographical coordinates directly
            return (
                tri.Triangulation(lon_nodes, lat_nodes, triangles=nv.T),
                (lon_nodes, lat_nodes),
                date_line_crossed,
            )

    def create_mesh_triangulation(self, ds, nodes_to_web_mercator=True):
        """Create a triangulation for mesh visualization."""
        lat_nodes = ds["lat_node"].values
        lon_nodes = ds["lon_node"].values
        nv = ds["nv"].values - 1

        if nodes_to_web_mercator:
            x_web, y_web = self.base.transformer_to_web_mercator.transform(
                lon_nodes, lat_nodes
            )
            return tri.Triangulation(x_web, y_web, triangles=nv.T), (x_web, y_web)
        else:
            return tri.Triangulation(lon_nodes, lat_nodes, triangles=nv.T), (
                lon_nodes,
                lat_nodes,
            )

    def add_basemap_to_visualization(self, ax, zoom_level):
        """Add a basemap to the visualization."""
        ctx.add_basemap(
            ax,
            source=self.basemap_provider,
            zoom=zoom_level,
            attribution=False,
            # verify=False,
            # timeout=30,
            # use_cache=False,
            # cache_dir="/scratch/asimms/Tidal/cache",
        )

    def render_tidal_data(self, ax, triang, data, variable_name):
        """Render tidal data on the provided axes."""
        # Create normalized colormap, using config-defined ranges if available
        if (
            variable_name in self.base.variable_ranges
            and "min" in self.base.variable_ranges[variable_name]
            and "max" in self.base.variable_ranges[variable_name]
        ):
            vmin = self.base.variable_ranges[variable_name]["min"]
            vmax = self.base.variable_ranges[variable_name]["max"]
            norm = Normalize(vmin=vmin, vmax=vmax)
        else:
            norm = Normalize(vmin=data.min(), vmax=data.max())

        # Plot the data
        tcf = ax.tripcolor(
            triang, facecolors=data, cmap=self.colormap, alpha=self.alpha, norm=norm
        )

        # Plot grid lines with configurable width and alpha
        ax.triplot(
            triang, color="black", lw=self.grid_line_width, alpha=self.grid_line_alpha
        )

        return tcf, norm

    def add_latlon_ticks(self, ax):
        """Add latitude and longitude tick marks to the axes."""
        x_ticks = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 6)
        x_labels = []
        for x in x_ticks:
            lon, _ = self.base.transformer_from_web_mercator.transform(x, 0)
            x_labels.append(f"{lon:.2f}°")
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels, fontsize=8 * self.font_scale)
        ax.set_xlabel("Longitude", fontsize=10 * self.font_scale)

        y_ticks = np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], 6)
        y_labels = []
        for y in y_ticks:
            _, lat = self.base.transformer_from_web_mercator.transform(0, y)
            y_labels.append(f"{lat:.2f}°")
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels, fontsize=8 * self.font_scale)
        ax.set_ylabel("Latitude", fontsize=10 * self.font_scale)

    def add_north_arrow(self, ax):
        """
        Add a north arrow to the visualization in the bottom right corner
        with proper alignment between the arrow and 'N' label.
        """
        # Define the position in axes coordinates (bottom right)
        base_x = 0.95
        base_y = 0.10

        # Arrow height
        arrow_height = 0.06

        # Create the arrow pointing north
        arrow_props = dict(
            arrowstyle="-|>",  # Changed to a cleaner arrowhead style
            lw=2.5,
            mutation_scale=15 * self.font_scale,
            color="white",
            joinstyle="miter",
            capstyle="butt",
        )

        # Create the arrow from bottom to top (north)
        north_arrow = FancyArrowPatch(
            (base_x, base_y),
            (base_x, base_y + arrow_height),
            transform=ax.transAxes,
            **arrow_props,
        )
        ax.add_patch(north_arrow)

        # Add the 'N' directly above the arrow
        ax.text(
            base_x,
            base_y + arrow_height + 0.01,  # Small gap between arrow and text
            "N",
            transform=ax.transAxes,
            ha="center",  # Horizontally centered
            va="bottom",  # Text sits above the anchor point
            color="white",
            fontweight="bold",
            fontsize=10 * self.font_scale,
        )

        # Optional: Add a small background box for better visibility
        # Uncomment if needed for better contrast against map background
        # box = Rectangle(
        #     (base_x - 0.03, base_y - 0.01),
        #     0.06, arrow_height + 0.07,
        #     transform=ax.transAxes,
        #     facecolor='black',
        #     alpha=0.3,
        #     zorder=9,
        #     edgecolor=None
        # )
        # ax.add_patch(box)

    def add_scale_bar(self, ax, x_min, x_max, y_min, y_max, use_metric=True):
        """
        Add a scale bar that is accurate based on the dataset dimensions.

        Parameters:
        -----------
        ax : matplotlib.axes.Axes
            The axes to add the scale bar to
        x_min, x_max, y_min, y_max : float
            The bounds of the current view in web mercator coordinates
        use_metric : bool, optional
            If True, use metric units (km, m). If False, use imperial units (mi, ft)
            Default is False (imperial/American units)

        Returns:
        --------
        scalebar : matplotlib_scalebar.scalebar.ScaleBar or None
            The scalebar object if successfully created, None otherwise
        """
        try:
            from matplotlib_scalebar.scalebar import ScaleBar

            # Get center point of the view
            center_x = (x_min + x_max) / 2
            center_y = (y_min + y_max) / 2

            # Convert center point to lat/lon
            center_lon, center_lat = self.base.transformer_from_web_mercator.transform(
                center_x, center_y
            )

            # Set dimension, units and fixed units based on user preference
            if use_metric:
                # For metric, use SI length
                dimension = "si-length"
                units = "m"  # Base unit must be meters for SI
                fixed_value = None  # Let the library determine the appropriate scale
                fixed_units = None
            else:
                # For imperial, use imperial length
                dimension = "imperial-length"
                units = "ft"  # Base unit must be feet for imperial
                fixed_value = None  # Let the library determine the appropriate scale
                # fixed_units = "mi"  # But ensure display is in miles
                fixed_units = None  # But ensure display is in miles

            # Create scale bar with appropriate units
            scalebar = ScaleBar(
                dx=1.0,  # Set to 1 as we'll use the built-in scaling
                units=units,
                dimension=dimension,
                color="white",
                box_alpha=0,
                location="lower left",
                border_pad=0.5,
                pad=0.5,
                rotation="horizontal-only",
                scale_loc="top",
                length_fraction=0.2,
                fixed_value=fixed_value,
                fixed_units=fixed_units,
            )

            # Add the scalebar to the axes
            ax.add_artist(scalebar)
            return scalebar
        except ImportError:
            print("matplotlib_scalebar not installed. Scale bar not added.")
            return None
        except Exception as e:
            print(
                f"Error adding scale bar: {str(e)}\nTrying fallback implementation..."
            )

            # Fallback implementation - try with more basic parameters
            try:
                if use_metric:
                    scalebar = ScaleBar(1, "m", dimension="si-length")
                else:
                    scalebar = ScaleBar(1, "ft", dimension="imperial-length")

                ax.add_artist(scalebar)
                return scalebar
            except Exception as e2:
                print(f"Fallback scale bar also failed: {str(e2)}")
                return None

    def save_visualization(self, fig, output_basename):
        """Save the visualization to file(s)."""
        saved_formats = []

        if "png" in self.output_formats:
            fig.savefig(
                f"{output_basename}.png",
                dpi=self.dpi,
                bbox_inches="tight",
                transparent=False,
                facecolor="white",
            )
            saved_formats.append(f"{output_basename}.png")

        if "pdf" in self.output_formats:
            fig.savefig(
                f"{output_basename}.pdf",
                dpi=self.dpi * 2,
                bbox_inches="tight",
                transparent=False,
                facecolor="white",
            )
            saved_formats.append(f"{output_basename}.pdf")

        if "svg" in self.output_formats:
            fig.savefig(
                f"{output_basename}.svg",
                bbox_inches="tight",
                transparent=False,
                facecolor="white",
            )
            saved_formats.append(f"{output_basename}.svg")

        if saved_formats:
            print(f"Figure saved as: {', '.join(saved_formats)}")

    def create_tidal_visualization(
        self,
        ds,
        variable_name,
        display_name=None,
        location_name="",
        layer_index=0,
        output_basename=None,
        zoom_level=10,
        bounds=None,
        units=None,
        cbar_min=None,
        cbar_max=None,
    ):
        """Create a visualization of tidal data with international date line handling."""
        # Setup fonts and plot environment
        self.setup_fonts()
        self.setup_visualization_environment()

        # Use variable name as display name if not provided
        if display_name is None:
            display_name = variable_name

        # Generate standardized output filename if needed
        if output_basename is None:
            if location_name:
                location_abbr = location_name.split(",")[0].lower().replace(" ", "_")
                output_basename = self.base.create_filename(
                    location_abbr, variable_name=variable_name
                )
            else:
                output_basename = f"tidal_{variable_name}_map"

        # Extract data using dedicated method
        print("Extracting variable data...")
        variable_data = get_variable_data(ds, variable_name, layer_index)

        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize, dpi=100)

        # Create triangulation with date line handling
        triang_web, (x_web, y_web), date_line_crossed = (
            self.create_mesh_triangulation_with_date_line_handling(ds)
        )

        # If date line was crossed, add a note to the plot title
        date_line_note = " (Crosses Int'l Date Line)" if date_line_crossed else ""

        # Plot the data
        print("Rendering tidal data...")
        tcf, norm = self.render_tidal_data(ax, triang_web, variable_data, variable_name)

        # Get units from dataset if not provided
        units = f"[{ds[variable_name].attrs.get('units', 'm/s')}]"
        long_name = f"{ds[variable_name].attrs.get('long_name', variable_name)}".title()
        short_name = f"{ds[variable_name].attrs.get('short_name', long_name)}".title()

        # Add colorbar
        print("Adding colorbar...")
        cbar = plt.colorbar(tcf, ax=ax, orientation="vertical", pad=0.02, shrink=0.6)
        cbar.ax.tick_params(labelsize=10 * self.font_scale)
        cbar.set_label(f"{short_name} {units}", fontsize=12 * self.font_scale)

        # if cbar_min is not None and cbar_max is not None:
        #     # cbar.set_clim(cbar_min, cbar_max)
        #     cbar.ax.set_ylim(cbar_min, cbar_max)

        if cbar_min is not None and cbar_max is not None:
            # Create a new normalization with the specified limits

            # Option 1: Use Normalize (linear mapping)
            # norm = mcolors.Normalize(vmin=cbar_min, vmax=cbar_max)

            # Option 2: Or if you want to clip values outside the range
            norm = mcolors.Normalize(vmin=cbar_min, vmax=cbar_max, clip=True)

            # Apply the normalization to the existing colormap
            tcf.set_norm(norm)

            # Update the colorbar
            cbar.update_normal(tcf)

        # Set plot bounds - either from provided bounds or from data with buffer
        if bounds:
            # If data crosses date line and bounds are provided, we need to check
            # if the bounds also need to be normalized
            if date_line_crossed:
                # Check if the bounds cross the date line
                has_negative_bound = bounds[0] < 0 or bounds[2] < 0
                has_positive_bound = bounds[0] > 0 or bounds[2] > 0
                bounds_span = abs(bounds[2] - bounds[0])

                if has_negative_bound and has_positive_bound and bounds_span > 180:
                    # Normalize the bounds to [0, 360] range
                    bounds_normalized = [
                        bounds[0] + 360 if bounds[0] < 0 else bounds[0],
                        bounds[1],
                        bounds[2] + 360 if bounds[2] < 0 else bounds[2],
                        bounds[3],
                    ]
                    bounds = bounds_normalized

            # Transform the geographic bounds to web mercator
            x_min, y_min = self.base.transformer_to_web_mercator.transform(
                bounds[0], bounds[1]
            )
            x_max, y_max = self.base.transformer_to_web_mercator.transform(
                bounds[2], bounds[3]
            )
        else:
            # Calculate from data with buffer
            x_min, x_max = min(x_web), max(x_web)
            y_min, y_max = min(y_web), max(y_web)

            # Add buffer around the bounds
            bounds_data = [x_min, y_min, x_max, y_max]
            buffered_bounds = self.base.add_buffer_to_bounds(bounds_data, 5)
            x_min, y_min, x_max, y_max = buffered_bounds

        # Apply bounds to plot
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        # Add latitude/longitude tick marks with date line handling
        if date_line_crossed:
            self.add_latlon_ticks_with_date_line_handling(ax)
        else:
            self.add_latlon_ticks(ax)

        # Add basemap
        print("Adding basemap...")
        self.add_basemap_to_visualization(ax, zoom_level)

        # Add title, including date line note if applicable
        plt.title(
            f"WPTO High Resolution Tidal Hindcast{date_line_note}\n{display_name}\n{long_name}",
            fontsize=16 * self.font_scale,
            pad=20,
        )

        self.add_scale_bar(ax, x_min, x_max, y_min, y_max)

        # Add north arrow
        self.add_north_arrow(ax)

        plt.tight_layout()

        # Save figure if configured to do so (not in Jupyter by default)
        if self.base.save_figures:
            self.save_visualization(fig, output_basename)

        # Always show in Jupyter
        if self.base.in_jupyter:
            plt.show()

        return fig, ax

    def add_latlon_ticks_with_date_line_handling(self, ax):
        """
        Add latitude and longitude tick marks to the axes with date line handling.
        This method handles normalized longitude values (0-360 range).
        """
        import numpy as np

        # Get the current x limits in web mercator coordinates
        x_min, x_max = ax.get_xlim()

        # Create evenly spaced ticks
        x_ticks = np.linspace(x_min, x_max, 6)
        x_labels = []

        for x in x_ticks:
            # Convert web mercator x back to longitude
            lon, _ = self.base.transformer_from_web_mercator.transform(x, 0)

            # Handle normalized longitudes (those > 180)
            if lon > 180:
                lon_display = lon - 360  # Convert back to -180 to 180 range for display
                x_labels.append(f"{lon_display:.2f}°")
            else:
                x_labels.append(f"{lon:.2f}°")

        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels, fontsize=8 * self.font_scale)
        ax.set_xlabel("Longitude", fontsize=10 * self.font_scale)

        # Latitude ticks (unchanged)
        y_ticks = np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], 6)
        y_labels = []
        for y in y_ticks:
            _, lat = self.base.transformer_from_web_mercator.transform(0, y)
            y_labels.append(f"{lat:.2f}°")
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels, fontsize=8 * self.font_scale)
        ax.set_ylabel("Latitude", fontsize=10 * self.font_scale)

    def visualize_area_of_interest(
        self,
        location_key,
        area_key,
        variable_name,
        display_name=None,
        layer_index=0,
        add_overview=True,
    ):
        """Visualize a specific area of interest."""
        # Get location and area information
        location_info = get_location_info(self.config, location_key)
        area_info = get_area_of_interest_info(self.config, location_key, area_key)

        # Find and load dataset
        dataset_path = find_dataset_file(
            self.base.base_data_dir,
            location_key,
            self.config["dirs"].get("end", ""),
        )
        ds = load_dataset(dataset_path)

        # Get units from dataset
        units = f"[{ds[variable_name].attrs.get('units', 'm/s')}]"

        # Create output filename
        output_basename = self.base.create_filename(
            location_key, area_key, variable_name
        )

        # Format display names
        location_display = format_location_display(location_info)
        area_display = format_area_of_interest_display(area_info)
        combined_display = format_combined_display(location_info, area_info)

        # Get bounds and determine zoom level
        bounds = area_info["bounds"]
        zoom_level = area_info.get(
            "zoom_level", self.base.determine_zoom_level_from_bounds(bounds)
        )

        # Create visualization
        fig, ax = self.create_tidal_visualization(
            ds=ds,
            variable_name=variable_name,
            display_name=display_name,
            location_name=combined_display,
            layer_index=layer_index,
            output_basename=output_basename,
            zoom_level=zoom_level,
            bounds=bounds,
            units=units,
        )

        # Add overview inset if requested
        if add_overview:
            inset_ax = self.add_overview_inset(
                fig, ax, location_key, area_key, dataset=ds
            )

        return fig, ax

    def add_overview_inset(self, fig, ax, location_key, area_key, dataset=None):
        """Add an overview inset to the visualization."""
        location_info = get_location_info(self.config, location_key)
        area_info = get_area_of_interest_info(self.config, location_key, area_key)

        inset_ax = inset_axes(
            ax, width="25%", height="25%", loc="upper right", borderpad=3
        )

        # Get all areas for this location
        all_area_bounds = [
            area["bounds"] for area in location_info["hotspots"].values()
        ]
        overview_bounds = self.base.calculate_combined_bounds(
            all_area_bounds, buffer_percent=20
        )

        area_bounds = area_info["bounds"]

        # Transform bounds to web mercator
        ov_x_min, ov_y_min = self.base.transformer_to_web_mercator.transform(
            overview_bounds[0], overview_bounds[1]
        )
        ov_x_max, ov_y_max = self.base.transformer_to_web_mercator.transform(
            overview_bounds[2], overview_bounds[3]
        )

        area_x_min, area_y_min = self.base.transformer_to_web_mercator.transform(
            area_bounds[0], area_bounds[1]
        )
        area_x_max, area_y_max = self.base.transformer_to_web_mercator.transform(
            area_bounds[2], area_bounds[3]
        )

        # Set inset axes limits
        inset_ax.set_xlim(ov_x_min, ov_x_max)
        inset_ax.set_ylim(ov_y_min, ov_y_max)

        # Draw rectangle highlighting current area
        rect_width = area_x_max - area_x_min
        rect_height = area_y_max - area_y_min

        rect = Rectangle(
            (area_x_min, area_y_min),
            rect_width,
            rect_height,
            linewidth=2,
            edgecolor="red",
            facecolor="none",
            zorder=10,
        )

        inset_ax.add_patch(rect)

        # Remove ticks
        inset_ax.set_xticks([])
        inset_ax.set_yticks([])
        inset_ax.set_xticklabels([])
        inset_ax.set_yticklabels([])

        # Set background and border
        inset_ax.set_facecolor("lightgray")
        for spine in inset_ax.spines.values():
            spine.set_edgecolor("black")
            spine.set_linewidth(1)

        return inset_ax

    def visualize_all_areas_of_interest(
        self,
        variable_name,
        display_name=None,
        selected_locations=None,
        selected_areas=None,
    ):
        """Visualize all areas of interest or a selected subset."""
        results = {}

        # Determine which locations to process
        if selected_locations is None:
            from .config import get_location_names

            locations_to_process = get_location_names(self.config)
        else:
            locations_to_process = selected_locations

        # Process each location
        for location_key in locations_to_process:
            # Determine which areas to process for this location
            if selected_areas and location_key in selected_areas:
                areas_to_process = selected_areas[location_key]
            else:
                from .config import get_areas_of_interest

                areas_to_process = get_areas_of_interest(self.config, location_key)

            # Process each area
            for area_key in areas_to_process:
                print(f"Visualizing {variable_name} for {location_key}/{area_key}")

                try:
                    fig, ax = self.visualize_area_of_interest(
                        location_key=location_key,
                        area_key=area_key,
                        variable_name=variable_name,
                        display_name=display_name,
                    )

                    results[f"{location_key}/{area_key}"] = (fig, ax)
                    plt.close(fig)
                except Exception as e:
                    print(f"Error visualizing {location_key}/{area_key}: {e}")

        return results

    def visualize_location_overview(
        self, location_key, variable_name, display_name=None, layer_index=0
    ):
        """Create an overview visualization for an entire location."""
        # Get location information
        location_info = get_location_info(self.config, location_key)

        # Find and load dataset
        dataset_path = find_dataset_file(
            self.base.base_data_dir,
            location_key,
            self.config["dirs"].get("end", ""),
        )
        ds = load_dataset(dataset_path)

        # Get units from dataset
        units = f"[{ds[variable_name].attrs.get('units', 'm/s')}]"

        # Create output filename
        output_basename = self.base.create_filename(
            location_key, suffix=f"overview_{variable_name}"
        )

        # Format display name
        location_display = format_location_display(location_info)

        # Calculate combined bounds from all areas
        all_bounds = [area["bounds"] for area in location_info["hotspots"].values()]
        overview_bounds = self.base.calculate_combined_bounds(
            all_bounds, buffer_percent=20
        )

        # Determine zoom level
        zoom_level = self.base.determine_zoom_level_from_bounds(overview_bounds)

        # Create visualization
        fig, ax = self.create_tidal_visualization(
            ds=ds,
            variable_name=variable_name,
            display_name=display_name,
            location_name=f"{location_display} Overview",
            layer_index=layer_index,
            output_basename=output_basename,
            zoom_level=zoom_level,
            bounds=overview_bounds,
            units=units,
        )

        # Add rectangles for each area of interest
        for area_key, area_info in location_info["hotspots"].items():
            bounds = area_info["bounds"]

            # Transform bounds to web mercator
            x_min, y_min = self.base.transformer_to_web_mercator.transform(
                bounds[0], bounds[1]
            )
            x_max, y_max = self.base.transformer_to_web_mercator.transform(
                bounds[2], bounds[3]
            )

            # Create rectangle
            rect = Rectangle(
                (x_min, y_min),
                x_max - x_min,
                y_max - y_min,
                linewidth=1.5,
                edgecolor="red",
                facecolor="none",
                zorder=10,
            )

            ax.add_patch(rect)

            # Add label at center of rectangle
            center_x = (x_min + x_max) / 2
            center_y = (y_min + y_max) / 2

            ax.text(
                center_x,
                center_y,
                area_info["display_name"],
                color="white",
                fontsize=8 * self.font_scale,
                ha="center",
                va="center",
                bbox=dict(facecolor="black", alpha=0.5, boxstyle="round,pad=0.3"),
                zorder=11,
            )

        return fig, ax

    def visualize_all_location_overviews(
        self, variable_name, display_name=None, selected_locations=None
    ):
        """Create overview visualizations for all locations or a selected subset."""
        results = {}

        # Determine which locations to process
        if selected_locations is None:
            from .config import get_location_names

            locations_to_process = get_location_names(self.config)
        else:
            locations_to_process = selected_locations

        # Process each location
        for location_key in locations_to_process:
            print(f"Creating overview for {location_key}")

            try:
                fig, ax = self.visualize_location_overview(
                    location_key=location_key,
                    variable_name=variable_name,
                    display_name=display_name,
                )

                results[location_key] = (fig, ax)
                plt.close(fig)
            except Exception as e:
                print(f"Error creating overview for {location_key}: {e}")

        return results
