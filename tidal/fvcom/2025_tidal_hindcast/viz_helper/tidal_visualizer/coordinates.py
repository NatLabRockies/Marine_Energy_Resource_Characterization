# coordinates module - Part of tidal_visualizer package

import numpy as np
from pyproj import Transformer, CRS


class CoordinateManager:
    """Manager for coordinate transformations in tidal visualizations."""

    def __init__(self):
        """Initialize the coordinate manager with common transformers."""
        # Cache for transformers to avoid recreating them
        self.transformers = {}

        # Initialize default transformers
        self._create_transformer("EPSG:4326", "EPSG:3857")  # WGS84 to Web Mercator
        self._create_transformer("EPSG:3857", "EPSG:4326")  # Web Mercator to WGS84

    def _create_transformer(self, from_crs, to_crs):
        """
        Create and cache a transformer between two CRS.

        Parameters:
        -----------
        from_crs : str
            Source CRS
        to_crs : str
            Target CRS

        Returns:
        --------
        pyproj.Transformer
            The transformer
        """
        key = f"{from_crs}->{to_crs}"
        if key not in self.transformers:
            self.transformers[key] = Transformer.from_crs(
                from_crs, to_crs, always_xy=True
            )
        return self.transformers[key]

    def transform_points(self, points, from_crs="EPSG:4326", to_crs="EPSG:3857"):
        """
        Transform points from one CRS to another.

        Parameters:
        -----------
        points : array-like
            Points to transform, as (x, y) pairs or separate x, y arrays
        from_crs : str, optional
            Source CRS (default: WGS84)
        to_crs : str, optional
            Target CRS (default: Web Mercator)

        Returns:
        --------
        numpy.ndarray
            Transformed points
        """
        # Get the transformer
        transformer = self._create_transformer(from_crs, to_crs)

        # Check if points is a tuple of x, y arrays
        if isinstance(points, tuple) and len(points) == 2:
            x, y = points
            x_trans, y_trans = transformer.transform(x, y)
            return np.array(x_trans), np.array(y_trans)

        # If it's a list/array of point tuples
        elif isinstance(points, (list, np.ndarray)):
            # Convert to separate x, y arrays
            points = np.asarray(points)
            if points.ndim == 2 and points.shape[1] >= 2:
                x, y = points[:, 0], points[:, 1]
                x_trans, y_trans = transformer.transform(x, y)

                # Combine back into point array
                result = np.column_stack((x_trans, y_trans))

                # If original points had more than 2 columns, preserve them
                if points.shape[1] > 2:
                    result = np.column_stack((result, points[:, 2:]))

                return result
            else:
                raise ValueError("Points must be an array of shape (n, 2) or more")
        else:
            raise ValueError("Unsupported points format")

    def transform_bounds(self, bounds, from_crs="EPSG:4326", to_crs="EPSG:3857"):
        """
        Transform bounds from one CRS to another.

        Parameters:
        -----------
        bounds : list-like
            Bounds as [xmin, ymin, xmax, ymax]
        from_crs : str, optional
            Source CRS
        to_crs : str, optional
            Target CRS

        Returns:
        --------
        list
            Transformed bounds
        """
        if len(bounds) != 4:
            raise ValueError(
                "Bounds must be a list or tuple of 4 values: [xmin, ymin, xmax, ymax]"
            )

        # Extract bounds
        xmin, ymin, xmax, ymax = bounds

        # Transform corners
        transformer = self._create_transformer(from_crs, to_crs)
        x1, y1 = transformer.transform(xmin, ymin)
        x2, y2 = transformer.transform(xmax, ymax)

        # Determine new bounds
        # Note: need to handle potential axis flips
        new_xmin, new_xmax = sorted([x1, x2])
        new_ymin, new_ymax = sorted([y1, y2])

        return [new_xmin, new_ymin, new_xmax, new_ymax]

    def get_utm_zone(self, lon, lat):
        """
        Calculate the UTM zone for a given longitude and latitude.

        Parameters:
        -----------
        lon : float
            Longitude
        lat : float
            Latitude

        Returns:
        --------
        int
            UTM zone number
        """
        # Special cases for Norway and Svalbard
        if 56 <= lat < 64 and 3 <= lon < 12:
            return 32
        elif 72 <= lat < 84 and 0 <= lon < 42:
            if lon < 9:
                return 31
            elif lon < 21:
                return 33
            elif lon < 33:
                return 35
            return 37

        # Standard case
        return int((lon + 180) / 6) + 1

    def get_utm_crs(self, lon, lat):
        """
        Get the UTM CRS for a given longitude and latitude.

        Parameters:
        -----------
        lon : float
            Longitude
        lat : float
            Latitude

        Returns:
        --------
        str
            UTM CRS string
        """
        zone = self.get_utm_zone(lon, lat)
        if lat >= 0:
            return f"EPSG:{32600 + zone}"  # Northern hemisphere
        else:
            return f"EPSG:{32700 + zone}"  # Southern hemisphere

    def create_utm_transformer(self, center_lon, center_lat):
        """
        Create a transformer from WGS84 to the appropriate UTM zone.

        Parameters:
        -----------
        center_lon : float
            Center longitude
        center_lat : float
            Center latitude

        Returns:
        --------
        pyproj.Transformer
            The transformer
        """
        utm_crs = self.get_utm_crs(center_lon, center_lat)
        return self._create_transformer("EPSG:4326", utm_crs)

    def get_transformer(self, from_crs, to_crs):
        """
        Get a transformer between two CRS.

        Parameters:
        -----------
        from_crs : str
            Source CRS
        to_crs : str
            Target CRS

        Returns:
        --------
        pyproj.Transformer
            The transformer
        """
        return self._create_transformer(from_crs, to_crs)

    def get_crs_info(self, crs_string):
        """
        Get information about a CRS.

        Parameters:
        -----------
        crs_string : str
            CRS string or EPSG code

        Returns:
        --------
        dict
            CRS information
        """
        crs = CRS.from_user_input(crs_string)

        # Extract useful information
        info = {
            "name": crs.name,
            "type": crs.type_name,
            "is_geographic": crs.is_geographic,
            "is_projected": crs.is_projected,
            "axis_info": [
                {"name": axis.name, "abbrev": axis.abbrev, "unit": axis.unit_name}
                for axis in crs.axis_info
            ],
        }

        # Add projection parameters if it's a projected CRS
        if crs.is_projected:
            info["datum"] = crs.datum.name if crs.datum else "Unknown"
            info["ellipsoid"] = crs.ellipsoid.name if crs.ellipsoid else "Unknown"
            info["units"] = crs.axis_info[0].unit_name if crs.axis_info else "Unknown"

        return info
