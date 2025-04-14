# core module - Part of tidal_visualizer package

import os
import numpy as np
from pyproj import Transformer


class TidalVisualizerBase:
    def __init__(self, visualization_config, base_data_dir=None):
        self.config = visualization_config

        # Set up base directory for data
        if base_data_dir is not None:
            self.base_data_dir = base_data_dir
        elif "dirs" in self.config and "base" in self.config["dirs"]:
            self.base_data_dir = self.config["dirs"]["base"]
        else:
            self.base_data_dir = os.getcwd()

        # Set up output directory
        if "dirs" in self.config and "output" in self.config["dirs"]:
            self.output_dir = self.config["dirs"]["output"]
        else:
            self.output_dir = os.path.join(self.base_data_dir, "output")

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

        # Detect if running in Jupyter
        self.in_jupyter = self._check_if_in_jupyter()
        self.save_figures = (
            not self.in_jupyter
        )  # Don't save figures by default in Jupyter

        # Create common transformers for reuse
        self.transformer_to_web_mercator = Transformer.from_crs(
            "EPSG:4326", "EPSG:3857", always_xy=True
        )
        self.transformer_from_web_mercator = Transformer.from_crs(
            "EPSG:3857", "EPSG:4326", always_xy=True
        )

        # Extract variable ranges from config if available
        self.variable_ranges = {}
        if "variables" in self.config and isinstance(self.config["variables"], dict):
            self.variable_ranges = self.config["variables"]

    def _check_if_in_jupyter(self):
        try:
            # Check if 'ipykernel' is in the sys.modules
            import sys

            return "ipykernel" in sys.modules
        except:
            return False

    def transform_coordinates(self, coords, from_crs="EPSG:4326", to_crs="EPSG:3857"):
        transformer = Transformer.from_crs(from_crs, to_crs, always_xy=True)

        if isinstance(coords, tuple) and len(coords) == 2:
            x_list, y_list = coords
            x_transformed, y_transformed = transformer.transform(x_list, y_list)
            return x_transformed, y_transformed
        else:
            transformed_coords = [transformer.transform(x, y) for x, y in coords]
            return transformed_coords

    def add_buffer_to_bounds(self, bounds, buffer_percent=10):
        # Check if these are web mercator bounds (4 values) or lat/lon bounds (4 values)
        if len(bounds) == 4:
            x_min, y_min, x_max, y_max = bounds

            x_range = x_max - x_min
            y_range = y_max - y_min

            x_buffer = x_range * (buffer_percent / 100)
            y_buffer = y_range * (buffer_percent / 100)

            return [
                x_min - x_buffer,
                y_min - y_buffer,
                x_max + x_buffer,
                y_max + y_buffer,
            ]
        else:
            raise ValueError(
                "Bounds must be a list or tuple of 4 values: [min_x, min_y, max_x, max_y]"
            )

    def calculate_bounds_from_points(self, points, buffer_percent=10):
        lons, lats = zip(*points)

        lon_min, lon_max = min(lons), max(lons)
        lat_min, lat_max = min(lats), max(lats)

        bounds = [lon_min, lat_min, lon_max, lat_max]

        if buffer_percent > 0:
            return self.add_buffer_to_bounds(bounds, buffer_percent)
        else:
            return bounds

    def determine_zoom_level_from_bounds(self, bounds):
        if bounds is not None:
            lon_min, lat_min, lon_max, lat_max = bounds

            lon_span = lon_max - lon_min

            if lon_span > 3:
                return 7
            elif lon_span > 1:
                return 8
            elif lon_span > 0.5:
                return 9
            elif lon_span > 0.2:
                return 10
            elif lon_span > 0.1:
                return 11
            elif lon_span > 0.05:
                return 12
            elif lon_span > 0.02:
                return 13
            else:
                return 14
        else:
            # Default zoom level if bounds not provided
            return 10

    def determine_zoom_level_from_coordinates(self, coordinates):
        if coordinates is not None:
            lons, lats = zip(*coordinates)
            lon_min, lon_max = min(lons), max(lons)
            lat_min, lat_max = min(lats), max(lats)

            # Now use the bounds determination
            return self.determine_zoom_level_from_bounds(
                [lon_min, lat_min, lon_max, lat_max]
            )
        else:
            # Default zoom level if coordinates not provided
            return 10

    def calculate_combined_bounds(self, bounds_list, buffer_percent=10):
        lon_min = min(bounds[0] for bounds in bounds_list)
        lat_min = min(bounds[1] for bounds in bounds_list)
        lon_max = max(bounds[2] for bounds in bounds_list)
        lat_max = max(bounds[3] for bounds in bounds_list)

        overall_bounds = [lon_min, lat_min, lon_max, lat_max]

        if buffer_percent > 0:
            return self.add_buffer_to_bounds(overall_bounds, buffer_percent)
        else:
            return overall_bounds

    def create_filename(
        self, location_key, area_of_interest=None, variable_name=None, suffix="map"
    ):
        location_abbr = location_key.lower().split("_")[-1]

        parts = ["tidal"]

        if location_abbr:
            parts.append(location_abbr)

        if area_of_interest:
            parts.append(area_of_interest)

        if variable_name:
            parts.append(variable_name)

        if suffix:
            parts.append(suffix)

        filename = "_".join(parts)
        return os.path.join(self.output_dir, filename)
