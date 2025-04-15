# context_layers module - Part of tidal_visualizer package

import os
import numpy as np
import matplotlib.pyplot as plt
import contextily as ctx
from matplotlib.patches import Polygon
from shapely.geometry import Point, LineString, shape
import geopandas as gpd

import ssl
import urllib3
import requests

from .utils import disable_ssl_verification


try:
    import cartopy.feature as cfeature
except ImportError:
    cfeature = None

# Dictionary of available basemaps
AVAILABLE_BASEMAPS = {
    "satellite": ctx.providers.Esri.WorldImagery,
    "topo": ctx.providers.Esri.WorldTopoMap,
    "streets": ctx.providers.OpenStreetMap.Mapnik,
    # "terrain": ctx.providers.Stamen.Terrain,
    # "nasa": ctx.providers.NASAGIBS.ModisTerraTrueColorCR,
    # "ocean": ctx.providers.Esri.OceanBasemap,
    # "gray": ctx.providers.Stamen.TonerLite,
}


class ContextLayerManager:
    """Manager for context layers in tidal visualizations."""

    def __init__(
        self,
        transformer_to_web_mercator,
        transformer_from_web_mercator,
        disable_ssl=True,
    ):
        self.transformer_to_web_mercator = transformer_to_web_mercator
        self.transformer_from_web_mercator = transformer_from_web_mercator
        self.layers = {}

        # Initialize with some standard layers
        self._initialize_standard_layers()

        if disable_ssl is True:
            disable_ssl_verification()

    def _initialize_standard_layers(self):
        """Initialize standard context layers if cartopy is available."""
        if cfeature is not None:
            self.layers.update(
                {
                    "coastline": cfeature.COASTLINE.scale("10m"),
                    "borders": cfeature.BORDERS.scale("10m"),
                    "land": cfeature.LAND.scale("10m"),
                    "ocean": cfeature.OCEAN.scale("10m"),
                    "lakes": cfeature.LAKES.scale("10m"),
                    "rivers": cfeature.RIVERS.scale("10m"),
                    "states": cfeature.STATES.scale("10m"),
                }
            )

    def add_basemap(self, ax, basemap="satellite", zoom=None, alpha=1.0):
        """
        Add a basemap to the axes.

        Parameters:
        -----------
        ax : matplotlib.axes.Axes
            Axes to add the basemap to
        basemap : str or ctx.providers object
            Name of the basemap or a contextily provider object
        zoom : int, optional
            Zoom level, if None will be determined automatically
        alpha : float, optional
            Transparency of the basemap

        Returns:
        --------
        None
        """
        # Get the basemap provider
        if isinstance(basemap, str) and basemap in AVAILABLE_BASEMAPS:
            provider = AVAILABLE_BASEMAPS[basemap]
        elif basemap in AVAILABLE_BASEMAPS.values():
            provider = basemap
        else:
            # Default to satellite imagery
            provider = AVAILABLE_BASEMAPS["satellite"]

        # Add the basemap
        ctx.add_basemap(ax, source=provider, zoom=zoom, alpha=alpha, attribution=False)

    def add_standard_layer(self, ax, layer_name, **kwargs):
        """
        Add a standard layer to the axes.

        Parameters:
        -----------
        ax : matplotlib.axes.Axes
            Axes to add the layer to
        layer_name : str
            Name of the layer to add
        **kwargs : dict
            Additional arguments to pass to ax.add_feature()

        Returns:
        --------
        matplotlib.artist.Artist or None
            The added feature or None if the layer was not found
        """
        if layer_name not in self.layers:
            print(
                f"Layer '{layer_name}' not found. Available layers: {list(self.layers.keys())}"
            )
            return None

        # Get default styling based on layer type
        default_style = self._get_default_style(layer_name)

        # Update with user provided kwargs
        default_style.update(kwargs)

        # Add the feature
        return ax.add_feature(self.layers[layer_name], **default_style)

    def _get_default_style(self, layer_name):
        """Get default styling for a layer based on its name."""
        if layer_name == "coastline":
            return {
                "edgecolor": "black",
                "facecolor": "none",
                "linewidth": 0.5,
                "zorder": 15,
            }
        elif layer_name == "borders":
            return {
                "edgecolor": "gray",
                "facecolor": "none",
                "linewidth": 0.5,
                "zorder": 15,
            }
        elif layer_name == "land":
            return {
                "edgecolor": "none",
                "facecolor": "lightgray",
                "alpha": 0.5,
                "zorder": 5,
            }
        elif layer_name == "ocean":
            return {
                "edgecolor": "none",
                "facecolor": "lightblue",
                "alpha": 0.3,
                "zorder": 5,
            }
        elif layer_name == "lakes":
            return {
                "edgecolor": "blue",
                "facecolor": "lightblue",
                "alpha": 0.5,
                "zorder": 10,
            }
        elif layer_name == "rivers":
            return {
                "edgecolor": "blue",
                "facecolor": "none",
                "linewidth": 0.5,
                "zorder": 10,
            }
        elif layer_name == "states":
            return {
                "edgecolor": "gray",
                "facecolor": "none",
                "linewidth": 0.3,
                "zorder": 10,
            }
        else:
            return {
                "edgecolor": "black",
                "facecolor": "none",
                "linewidth": 0.5,
                "zorder": 10,
            }

    def add_shapefile(
        self,
        ax,
        shapefile_path,
        style=None,
        label_field=None,
        label_kwargs=None,
        transform_to_web_mercator=True,
    ):
        """
        Add a shapefile as a context layer.

        Parameters:
        -----------
        ax : matplotlib.axes.Axes
            Axes to add the layer to
        shapefile_path : str
            Path to the shapefile
        style : dict or callable, optional
            Style parameters for the shapes or a function that takes a shape and returns style parameters
        label_field : str, optional
            Field to use for labeling features
        label_kwargs : dict, optional
            Styling options for labels
        transform_to_web_mercator : bool, optional
            Whether to transform coordinates to web mercator

        Returns:
        --------
        list
            List of added shape objects
        """
        # Check if geopandas is available
        if gpd is None:
            print("GeoPandas is required to add shapefiles but is not installed.")
            return []

        # Read the shapefile
        try:
            gdf = gpd.read_file(shapefile_path)
        except Exception as e:
            print(f"Error reading shapefile: {e}")
            return []

        # Default style
        default_style = {
            "edgecolor": "black",
            "facecolor": "none",
            "linewidth": 0.5,
            "alpha": 0.7,
            "zorder": 10,
        }

        # Update with user provided style
        if style and isinstance(style, dict):
            default_style.update(style)

        # Default label style
        default_label_kwargs = {
            "fontsize": 8,
            "ha": "center",
            "va": "center",
            "color": "black",
            "bbox": dict(facecolor="white", alpha=0.7, pad=0.5, boxstyle="round"),
            "zorder": 15,
        }

        # Update with user provided label style
        if label_kwargs and isinstance(label_kwargs, dict):
            default_label_kwargs.update(label_kwargs)

        # List to store added shapes
        added_shapes = []

        # Process each shape
        for idx, row in gdf.iterrows():
            geom = row.geometry

            # Get style for this shape
            if callable(style):
                shape_style = style(row)
                # Update default style with shape-specific style
                current_style = default_style.copy()
                current_style.update(shape_style)
            else:
                current_style = default_style

            # Convert to shapely geometry if it's not already
            if hasattr(geom, "__geo_interface__"):
                geom = shape(geom.__geo_interface__)

            # Handle different geometry types
            if transform_to_web_mercator:
                # For polygons
                if geom.geom_type == "Polygon":
                    # Transform coordinates
                    x, y = geom.exterior.xy
                    x_mercator, y_mercator = self.transformer_to_web_mercator.transform(
                        x, y
                    )

                    # Create and add the polygon
                    poly = Polygon(
                        np.column_stack([x_mercator, y_mercator]), **current_style
                    )
                    ax.add_patch(poly)
                    added_shapes.append(poly)

                # For lines
                elif geom.geom_type == "LineString":
                    x, y = geom.xy
                    x_mercator, y_mercator = self.transformer_to_web_mercator.transform(
                        x, y
                    )
                    (line,) = ax.plot(x_mercator, y_mercator, **current_style)
                    added_shapes.append(line)

                # For points
                elif geom.geom_type == "Point":
                    x, y = geom.x, geom.y
                    x_mercator, y_mercator = self.transformer_to_web_mercator.transform(
                        [x], [y]
                    )
                    point = ax.scatter(x_mercator, y_mercator, **current_style)
                    added_shapes.append(point)

                # For collections (MultiPolygon, etc.)
                elif "Multi" in geom.geom_type:
                    for part in geom.geoms:
                        if part.geom_type == "Polygon":
                            x, y = part.exterior.xy
                            x_mercator, y_mercator = (
                                self.transformer_to_web_mercator.transform(x, y)
                            )
                            poly = Polygon(
                                np.column_stack([x_mercator, y_mercator]),
                                **current_style,
                            )
                            ax.add_patch(poly)
                            added_shapes.append(poly)
                        elif part.geom_type == "LineString":
                            x, y = part.xy
                            x_mercator, y_mercator = (
                                self.transformer_to_web_mercator.transform(x, y)
                            )
                            (line,) = ax.plot(x_mercator, y_mercator, **current_style)
                            added_shapes.append(line)
                        elif part.geom_type == "Point":
                            x, y = part.x, part.y
                            x_mercator, y_mercator = (
                                self.transformer_to_web_mercator.transform([x], [y])
                            )
                            point = ax.scatter(x_mercator, y_mercator, **current_style)
                            added_shapes.append(point)
            else:
                # If not transforming to web mercator, just add the shape directly
                # (assuming ax is already in the right projection)
                if geom.geom_type == "Polygon":
                    x, y = geom.exterior.xy
                    poly = Polygon(np.column_stack([x, y]), **current_style)
                    ax.add_patch(poly)
                    added_shapes.append(poly)
                elif geom.geom_type == "LineString":
                    x, y = geom.xy
                    (line,) = ax.plot(x, y, **current_style)
                    added_shapes.append(line)
                elif geom.geom_type == "Point":
                    x, y = geom.x, geom.y
                    point = ax.scatter(x, y, **current_style)
                    added_shapes.append(point)
                elif "Multi" in geom.geom_type:
                    for part in geom.geoms:
                        if part.geom_type == "Polygon":
                            x, y = part.exterior.xy
                            poly = Polygon(np.column_stack([x, y]), **current_style)
                            ax.add_patch(poly)
                            added_shapes.append(poly)
                        elif part.geom_type == "LineString":
                            x, y = part.xy
                            (line,) = ax.plot(x, y, **current_style)
                            added_shapes.append(line)
                        elif part.geom_type == "Point":
                            x, y = part.x, part.y
                            point = ax.scatter(x, y, **current_style)
                            added_shapes.append(point)

            # Add label if requested
            if label_field and label_field in row:
                # Get label text
                label_text = str(row[label_field])

                # Get centroid for label placement
                if geom.geom_type == "Polygon" or geom.geom_type == "MultiPolygon":
                    centroid = geom.centroid
                    cx, cy = centroid.x, centroid.y
                elif (
                    geom.geom_type == "LineString"
                    or geom.geom_type == "MultiLineString"
                ):
                    # Use midpoint of the line for label
                    if geom.geom_type == "LineString":
                        midpoint = geom.interpolate(0.5, normalized=True)
                    else:
                        # For MultiLineString, use the longest part
                        longest_part = max(geom.geoms, key=lambda g: g.length)
                        midpoint = longest_part.interpolate(0.5, normalized=True)
                    cx, cy = midpoint.x, midpoint.y
                else:  # Point or MultiPoint
                    if geom.geom_type == "Point":
                        cx, cy = geom.x, geom.y
                    else:
                        # For MultiPoint, use the first point
                        cx, cy = geom.geoms[0].x, geom.geoms[0].y

                # Transform to web mercator if needed
                if transform_to_web_mercator:
                    cx_mercator, cy_mercator = (
                        self.transformer_to_web_mercator.transform([cx], [cy])
                    )
                    cx, cy = cx_mercator[0], cy_mercator[0]

                # Add the label
                label = ax.text(cx, cy, label_text, **default_label_kwargs)
                added_shapes.append(label)

        return added_shapes

    def add_navigation_channels(self, ax, shapefile_path=None):
        """
        Add navigation channels as a context layer.

        Parameters:
        -----------
        ax : matplotlib.axes.Axes
            Axes to add the layer to
        shapefile_path : str, optional
            Path to the navigation channels shapefile

        Returns:
        --------
        list
            List of added shape objects
        """
        # Default style for navigation channels
        nav_style = {
            "edgecolor": "navy",
            "facecolor": "lightblue",
            "alpha": 0.6,
            "linewidth": 1.0,
            "zorder": 8,
            "linestyle": "--",
        }

        # Label style for navigation channels
        label_kwargs = {
            "fontsize": 8,
            "ha": "center",
            "va": "center",
            "color": "navy",
            "bbox": dict(facecolor="white", alpha=0.7, pad=0.5, boxstyle="round"),
            "zorder": 15,
        }

        # If shapefile path is provided, add it
        # if shapefile_path and os.path.exists(shapefile_path):
        #     return self.add_shapefile(
        #         ax,
        #         shapefile_path,
        #         style=style_by_type,
        #         label_field="NAME",  # Assuming there's a NAME field in the shapefile
        #         label_kwargs=label_kwargs,
        #     )

        # If no shapefile path is provided, return empty list
        return []

    def add_marine_protected_areas(self, ax, shapefile_path=None):
        """
        Add marine protected areas as a context layer.

        Parameters:
        -----------
        ax : matplotlib.axes.Axes
            Axes to add the layer to
        shapefile_path : str, optional
            Path to the marine protected areas shapefile

        Returns:
        --------
        list
            List of added shape objects
        """
        # Default style for marine protected areas
        mpa_style = {
            "edgecolor": "green",
            "facecolor": "lightgreen",
            "alpha": 0.4,
            "linewidth": 0.8,
            "zorder": 9,
            "linestyle": "-.",
        }

        # Label style for marine protected areas
        label_kwargs = {
            "fontsize": 8,
            "ha": "center",
            "va": "center",
            "color": "darkgreen",
            "bbox": dict(facecolor="white", alpha=0.7, pad=0.5, boxstyle="round"),
            "zorder": 15,
        }

        # If shapefile path is provided, add it
        if shapefile_path and os.path.exists(shapefile_path):
            return self.add_shapefile(
                ax,
                shapefile_path,
                style=mpa_style,
                label_field="NAME",  # Assuming there's a NAME field in the shapefile
                label_kwargs=label_kwargs,
            )

        # If no shapefile path is provided, return empty list
        return []

    def add_bathymetry_contours(
        self,
        ax,
        ds,
        contour_levels=10,
        colors="blue",
        alpha=0.7,
        linewidths=0.5,
        zorder=6,
        labels=True,
    ):
        """
        Add bathymetry contour lines.

        Parameters:
        -----------
        ax : matplotlib.axes.Axes
            Axes to add the contours to
        ds : xarray.Dataset
            Dataset containing bathymetry data
        contour_levels : int or array-like, optional
            Number of contour levels or explicit list of levels
        colors : str or sequence, optional
            Colors for contour lines
        alpha : float, optional
            Transparency for contour lines
        linewidths : float or sequence, optional
            Line widths for contour lines
        zorder : int, optional
            Z-order for contour lines
        labels : bool, optional
            Whether to add contour labels

        Returns:
        --------
        matplotlib.contour.QuadContourSet
            The contour set
        """
        # Extract bathymetry data
        if "seafloor_depth" in ds:
            depth_var = "seafloor_depth"
        elif "depth" in ds:
            depth_var = "depth"
        elif "bathymetry" in ds:
            depth_var = "bathymetry"
        else:
            print("No bathymetry or depth variable found in dataset")
            return None

        depths = ds[depth_var].values
        lats = ds.lat_center.values
        lons = ds.lon_center.values

        # Transform coordinates to web mercator
        x, y = self.transformer_to_web_mercator.transform(lons, lats)

        # Create contour levels
        if isinstance(contour_levels, int):
            min_depth = np.min(depths)
            max_depth = np.max(depths)
            levels = np.linspace(min_depth, max_depth, contour_levels)
        else:
            levels = contour_levels

        # Create contours
        contours = ax.tricontour(
            x,
            y,
            depths,
            levels=levels,
            colors=colors,
            alpha=alpha,
            linewidths=linewidths,
            zorder=zorder,
        )

        # Add contour labels if requested
        if labels:
            # Format labels to show depth in meters
            fmt = lambda x: f"{x:.0f}m"

            # Add the labels
            ax.clabel(contours, inline=True, fontsize=8, fmt=fmt, colors="black")

        # return contours.path.exists(shapefile_path):
        #     return self.add_shapefile(
        #         ax,
        #         shapefile_path,
        #         style=nav_style,
        #         label_field='NAME',  # Assuming there's a NAME field in the shapefile
        #         label_kwargs=label_kwargs
        #     )

        # If no shapefile path is provided, return empty list
        return []

    def add_marine_infrastructure(self, ax, shapefile_path=None):
        """
        Add marine infrastructure as a context layer (ports, bridges, etc.).

        Parameters:
        -----------
        ax : matplotlib.axes.Axes
            Axes to add the layer to
        shapefile_path : str, optional
            Path to the marine infrastructure shapefile

        Returns:
        --------
        list
            List of added shape objects
        """
        # Default style for marine infrastructure
        infra_style = {
            "edgecolor": "black",
            "facecolor": "gray",
            "alpha": 0.7,
            "linewidth": 1.0,
            "zorder": 12,
        }

        # Function to determine style based on infrastructure type
        def style_by_type(feature):
            infra_type = feature.get("TYPE", "").lower()

            if "port" in infra_type or "harbor" in infra_type:
                return {
                    "edgecolor": "darkred",
                    "facecolor": "red",
                    "alpha": 0.6,
                    "linewidth": 1.0,
                    "zorder": 12,
                }
            elif "bridge" in infra_type:
                return {
                    "edgecolor": "black",
                    "facecolor": "none",
                    "alpha": 0.8,
                    "linewidth": 2.0,
                    "zorder": 12,
                }
            elif "dam" in infra_type:
                return {
                    "edgecolor": "black",
                    "facecolor": "darkgray",
                    "alpha": 0.8,
                    "linewidth": 1.5,
                    "zorder": 12,
                }
            else:
                return infra_style

        # Label style for marine infrastructure
        label_kwargs = {
            "fontsize": 8,
            "ha": "center",
            "va": "center",
            "color": "black",
            "bbox": dict(facecolor="white", alpha=0.7, pad=0.5, boxstyle="round"),
            "zorder": 15,
        }

        # If shapefile path is provided, add it
        # if shapefile_path and os
