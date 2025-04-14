import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

from matplotlib.patches import Polygon
from shapely.geometry import Point, LineString, Polygon as ShapelyPolygon
from scipy.spatial import cKDTree


def convert_xarray_to_dataframe(ds):
    face_vars = []
    sigma_vars = []
    other_vars = []

    for var_name in ds.data_vars:
        dims = ds[var_name].dims
        if len(dims) == 1 and "face" in dims:
            face_vars.append(var_name)
        elif len(dims) == 2 and "sigma_layer" in dims and "face" in dims:
            sigma_vars.append(var_name)
        else:
            other_vars.append(var_name)

    print(f"Face-only variables: {len(face_vars)}")
    print(f"Sigma layer variables: {len(sigma_vars)}")
    print(f"Other variables: {len(other_vars)}")

    print("Creating dataframe with all variables...")

    # Create a dictionary of all data we want to include
    data_dict = {
        "lat_center": ds.lat_center.values,
        "lon_center": ds.lon_center.values,
    }

    # Add element corner coordinates
    print("Adding element corner coordinates...")
    nv = ds.nv.values.T - 1  # Adjust for 0-based indexing if needed

    # Extract corner coordinates
    for i in range(3):  # Each element has 3 corners
        # Get node indices for this corner
        corner_indices = nv[:, i]

        # Add lat/lon for each corner
        data_dict[f"element_corner_{i+1}_lat"] = ds.lat_node.values[corner_indices]
        data_dict[f"element_corner_{i+1}_lon"] = ds.lon_node.values[corner_indices]

    # Add face-only variables
    if face_vars:
        print(f"Adding {len(face_vars)} face-only variables...")
        for var_name in face_vars:
            data_dict[var_name] = ds[var_name].values

    # Add sigma layer variables with suffixes
    if sigma_vars:
        print(
            f"Adding {len(sigma_vars)} sigma layer variables with sigma level suffixes..."
        )
        n_sigma = len(ds.sigma_layer)

        for var_name in sigma_vars:
            for sigma_idx in range(n_sigma):
                sigma_level = sigma_idx + 1
                column_name = f"{var_name}_sigma_level_{sigma_level}"
                data_dict[column_name] = ds[var_name].values[sigma_idx, :]

    # Create dataframe all at once
    result_df = pd.DataFrame(data_dict)

    print(f"Created dataframe with {result_df.shape[1]} columns")

    return result_df


def plot_vertical_transect(
    df,
    start_point,
    direction_deg,
    max_distance_km=10,
    variable="speed",
    auto_shore_detect=True,
    max_elements_warning=10000,
    bidirectional=True,
    plot_type="surface",
):
    """
    Plot a vertical transect from a starting point in both directions (forward and opposite),
    finding triangular elements that the line passes through and detecting shores automatically.

    Args:
        df: DataFrame with tidal data
        start_point: tuple of (lat, lon) for starting point
        direction_deg: direction in degrees (0 = North, 90 = East, etc.)
        max_distance_km: maximum distance to search in each direction (in km), None for default (50km)
        variable: variable to plot ('speed', 'power_density', 'volume_energy_flux')
        auto_shore_detect: whether to automatically detect and stop at shorelines
        max_elements_warning: warn if number of elements exceeds this threshold
        bidirectional: whether to search in both directions from start_point
        plot_type: 'surface' for filled contour visualization, 'scatter' for discrete points
    """
    print(
        f"Starting {'bi-directional' if bidirectional else ''} transect plot from {start_point} in direction {direction_deg}°"
    )
    print(f"Variable selected: {variable}")
    print(f"Plot type: {plot_type}")

    # Convert direction to radians
    direction_rad = np.radians(direction_deg)

    # Create unit vector for the forward direction
    forward_unit = np.array([np.sin(direction_rad), np.cos(direction_rad)])

    # If bidirectional, we'll also create a opposite direction vector
    if bidirectional:
        opposite_unit = -forward_unit
        print("Searching in both forward and opposite directions from start point")

    # Approximate conversion from km to degrees
    km_per_deg_lat = 111.0
    km_per_deg_lon = 111.0 * np.cos(np.radians(start_point[0]))

    # Calculate endpoints for the search
    # Forward endpoint
    forward_delta_lat = max_distance_km * forward_unit[1] / km_per_deg_lat
    forward_delta_lon = max_distance_km * forward_unit[0] / km_per_deg_lon
    forward_end = (
        start_point[0] + forward_delta_lat,
        start_point[1] + forward_delta_lon,
    )

    # Opposite endpoint (if bidirectional)
    if bidirectional:
        opposite_delta_lat = max_distance_km * opposite_unit[1] / km_per_deg_lat
        opposite_delta_lon = max_distance_km * opposite_unit[0] / km_per_deg_lon
        opposite_end = (
            start_point[0] + opposite_delta_lat,
            start_point[1] + opposite_delta_lon,
        )
        print(f"Forward search end point: {forward_end} (at {max_distance_km} km)")
        print(f"Opposite search end point: {opposite_end} (at {max_distance_km} km)")
    else:
        print(f"Search end point: {forward_end} (at {max_distance_km} km)")

    # Create lines for searching
    forward_line = LineString(
        [start_point[::-1], forward_end[::-1]]
    )  # Shapely uses (lon, lat)
    if bidirectional:
        opposite_line = LineString([start_point[::-1], opposite_end[::-1]])

    # Build a tree for efficient lat/lon searching
    # Extract center points of all elements for KD-Tree
    element_centers = df[["lon_center", "lat_center"]].values
    kdtree = cKDTree(element_centers)

    # Define a buffer distance around the transect line (in degrees)
    buffer_dist_km = 2.0  # 2 km buffer around line
    buffer_dist_deg = buffer_dist_km / km_per_deg_lat  # rough conversion to degrees

    # --- Find elements along forward direction ---
    print("Searching for elements in forward direction...")
    forward_elements, forward_distances = find_elements_along_line(
        df,
        kdtree,
        forward_line,
        start_point,
        buffer_dist_deg,
        km_per_deg_lat,
        km_per_deg_lon,
    )

    # --- If bidirectional, find elements in opposite direction ---
    if bidirectional:
        print("Searching for elements in opposite direction...")
        opposite_elements, opposite_distances = find_elements_along_line(
            df,
            kdtree,
            opposite_line,
            start_point,
            buffer_dist_deg,
            km_per_deg_lat,
            km_per_deg_lon,
        )
        # Convert opposite distances to negative values to represent opposite direction
        opposite_distances = [-d for d in opposite_distances]

    # --- SHORE DETECTION ---
    if auto_shore_detect:
        print("Performing automatic shore detection...")

        # Process forward direction
        if len(forward_elements) > 0:
            print("Analyzing forward direction for shore boundaries...")
            forward_elements, forward_distances = detect_shore(
                forward_elements, forward_distances
            )

        # Process opposite direction if bidirectional
        if bidirectional and len(opposite_elements) > 0:
            print("Analyzing opposite direction for shore boundaries...")
            opposite_elements, opposite_distances = detect_shore(
                opposite_elements, opposite_distances
            )

    # --- Combine forward and opposite directions (if bidirectional) ---
    if bidirectional:
        # Merge and sort the two lists (opposite elements should come first since they have negative distances)
        combined_elements = opposite_elements + forward_elements
        combined_distances = opposite_distances + forward_distances

        # Sort by distance (this will automatically place opposite direction elements first)
        sorted_pairs = sorted(zip(combined_distances, combined_elements))
        distances = [dist for dist, _ in sorted_pairs]
        intersecting_elements = [idx for _, idx in sorted_pairs]
    else:
        # Only forward direction
        intersecting_elements = forward_elements
        distances = forward_distances

    print(f"Total elements along transect: {len(intersecting_elements)}")

    # --- Check if we need to warn about too many elements ---
    if len(intersecting_elements) > max_elements_warning:
        print(
            f"WARNING: Transect contains {len(intersecting_elements)} elements, which exceeds the warning threshold of {max_elements_warning}."
        )
        print("This may cause performance issues or produce a cluttered plot.")
        print(
            "Consider reducing max_distance_km or choosing a different transect direction."
        )

    if len(intersecting_elements) == 0:
        print("No intersecting elements found. Try adjusting start point or direction.")
        return None, None

    # Create a subset with only the intersecting elements
    df_transect = df.iloc[intersecting_elements].copy()
    df_transect["distance_along_transect"] = distances

    # Get depths for all sigma levels
    depth_cols = [col for col in df_transect.columns if "depth_sigma_level" in col]
    sigma_levels = list(range(1, len(depth_cols) + 1))

    # Extract the variable data for all sigma levels
    valid_variables = ["speed", "power_density", "volume_energy_flux"]
    if variable not in valid_variables:
        print(
            f"Warning: '{variable}' is not in the list of valid variables {valid_variables}"
        )
        print("Defaulting to 'speed'")
        variable = "speed"

    if variable == "speed":
        var_cols = [f"speed_sigma_level_{i}" for i in sigma_levels]
    elif variable == "power_density":
        var_cols = [f"power_density_sigma_level_{i}" for i in sigma_levels]
    elif variable == "volume_energy_flux":
        var_cols = [f"volume_energy_flux_sigma_level_{i}" for i in sigma_levels]

    print(f"Plotting {variable} across {len(sigma_levels)} sigma levels")

    # --- Create the plot ---
    fig, ax = plt.subplots(figsize=(14, 8))

    # Find global min/max for consistent color scale
    all_values = []
    for _, row in df_transect.iterrows():
        values = [row[f"{variable}_sigma_level_{j}"] for j in sigma_levels]
        all_values.extend(values)

    vmin, vmax = np.nanmin(all_values), np.nanmax(all_values)
    print(f"Value range for {variable}: {vmin:.3f} to {vmax:.3f}")

    # Create arrays to hold the data for the filled contour plot
    # We'll create a 2D grid of distances and depths
    x_positions = df_transect["distance_along_transect"].values
    depths_2d = []
    values_2d = []

    # Extract depth levels and values for each element along the transect
    for i, (_, row) in enumerate(df_transect.iterrows()):
        # Get depths for each sigma level
        depths = [row[f"depth_sigma_level_{j}"] for j in sigma_levels]
        depths_2d.append(depths)

        # Get values for each sigma level
        values = [row[f"{variable}_sigma_level_{j}"] for j in sigma_levels]
        values_2d.append(values)

    # Convert to numpy arrays
    depths_2d = np.array(depths_2d)
    values_2d = np.array(values_2d)

    print(f"Created 2D grid with shape: {depths_2d.shape}")

    # Use pcolormesh for a filled surface visualization
    if len(x_positions) > 1:
        print("len x_positions is > 1")
        # To use pcolormesh properly, we need to create meshgrid of cell edges
        # First, create edges for the horizontal axis (distances)
        x_edges = np.zeros(len(x_positions) + 1)
        for i in range(1, len(x_positions)):
            x_edges[i] = (x_positions[i - 1] + x_positions[i]) / 2
        # Handle first and last edges
        if len(x_positions) > 1:
            x_edges[0] = x_positions[0] - (x_positions[1] - x_positions[0]) / 2
            x_edges[-1] = x_positions[-1] + (x_positions[-1] - x_positions[-2]) / 2
        else:
            x_edges[0] = x_positions[0] - 0.5
            x_edges[-1] = x_positions[0] + 0.5

        # Now create edges for the vertical axis (depths)
        # For simplicity, we'll use the sigma level depths directly
        # This assumes the sigma levels represent the cell centers
        y_edges = np.zeros((depths_2d.shape[0], depths_2d.shape[1] + 1))

        for i in range(depths_2d.shape[0]):
            depths = depths_2d[i]
            # Calculate edges as midpoints between levels
            for j in range(1, len(depths)):
                y_edges[i, j] = (depths[j - 1] + depths[j]) / 2
            # Handle top and bottom edges
            if len(depths) > 1:
                y_edges[i, 0] = depths[0] - (depths[1] - depths[0]) / 2
                y_edges[i, -1] = depths[-1] + (depths[-1] - depths[-2]) / 2
            else:
                y_edges[i, 0] = depths[0] - 0.5
                y_edges[i, -1] = depths[0] + 0.5

        # Now create the pcolormesh plot
        # We need to loop through each column and create a quad patch
        for i in range(len(x_positions)):
            for j in range(len(sigma_levels)):
                # if j < len(sigma_levels) - 1:  # Skip the last level
                # Create a polygon for each cell
                x_quad = [x_edges[i], x_edges[i + 1], x_edges[i + 1], x_edges[i]]
                y_quad = [
                    y_edges[i, j],
                    y_edges[i, j],
                    y_edges[i, j + 1],
                    y_edges[i, j + 1],
                ]

                # Color based on the value at this point
                value = values_2d[i, j]

                # Normalize the value for colormapping
                norm_value = (value - vmin) / (vmax - vmin) if vmax > vmin else 0.5
                # Get color from colormap
                color = plt.cm.viridis(norm_value)

                # Add polygon patch
                poly = plt.Polygon(
                    np.array([x_quad, y_quad]).T,
                    facecolor=color,
                    edgecolor="none",
                    alpha=0.8,
                )
                ax.add_patch(poly)

        # Create a dummy scatter plot with the correct colormap for the colorbar
        sc = ax.scatter([], [], c=[], cmap="viridis", vmin=vmin, vmax=vmax, s=0)

        # Also add scatter points for the actual data points (optional)
        # for i, x_pos in enumerate(x_positions):
        #     ax.scatter(
        #         [x_pos] * len(sigma_levels),
        #         depths_2d[i],
        #         c=values_2d[i],
        #         cmap="viridis",
        #         s=10,
        #         vmin=vmin,
        #         vmax=vmax,
        #         edgecolor="gray",
        #         linewidth=0.5,
        #     )
    else:
        # If we only have one position, just use scatter plot
        x_pos = x_positions[0]
        depths = depths_2d[0]
        values = values_2d[0]
        sc = ax.scatter(
            [x_pos] * len(depths),
            depths,
            c=values,
            cmap="viridis",
            s=30,
            vmin=vmin,
            vmax=vmax,
        )
        ax.plot([x_pos] * len(depths), depths, "-", color="lightgray", alpha=0.5)

    # Add vertical line at starting point (distance=0)
    ax.axvline(x=0, color="red", linestyle="--", linewidth=1, alpha=0.7)
    ax.text(
        0, ax.get_ylim()[0], "Start", color="red", ha="center", va="bottom", fontsize=9
    )

    # Configure the plot
    cbar = plt.colorbar(sc, label=f'{variable.replace("_", " ").title()}')
    ax.set_ylabel("Depth (m)")
    ax.set_xlabel("Distance along transect (km)")

    # Update title to include actual transect length
    total_length = distances[-1] - distances[0] if distances else 0
    ax.set_title(
        f"Vertical Transect from ({start_point[0]:.4f}, {start_point[1]:.4f}) "
        + (
            f"in directions {direction_deg}° and {(direction_deg+180)%360}°\n"
            if bidirectional
            else f"in direction {direction_deg}°\n"
        )
        + f"Length: {abs(total_length):.2f} km, Elements: {len(intersecting_elements)}"
    )
    depth_line = df_transect["seafloor_depth"].to_numpy() * -1
    ax.plot(df_transect["distance_along_transect"], depth_line, color="black")

    ax.invert_yaxis()  # Invert y axis so depth increases downward

    # ax.plot(df_transect["seafloor_depth"] * -1.0, color="black", linewidth=0.75)

    # --- Add a map showing the transect line ---
    # if len(df_transect) > 0:
    #     # Calculate actual endpoints of the detected transect
    #     first_element = df_transect.iloc[0]
    #     last_element = df_transect.iloc[-1]
    #     actual_first = (first_element["lat_center"], first_element["lon_center"])
    #     actual_last = (last_element["lat_center"], last_element["lon_center"])
    #
    #     # Create an inset axis for the map
    #     axins = fig.add_axes([0.15, 0.15, 0.3, 0.3])
    #
    #     # Plot a limited number of triangles as context (to avoid performance issues)
    #     max_triangles_to_plot = 300
    #     context_indices = np.random.choice(
    #         len(df), min(max_triangles_to_plot, len(df)), replace=False
    #     )
    #
    #     for idx in context_indices:
    #         row = df.iloc[idx]
    #         triangle = Polygon(
    #             [
    #                 (row["element_corner_1_lon"], row["element_corner_1_lat"]),
    #                 (row["element_corner_2_lon"], row["element_corner_2_lat"]),
    #                 (row["element_corner_3_lon"], row["element_corner_3_lat"]),
    #             ],
    #             fill=False,
    #             edgecolor="lightgray",
    #             linewidth=0.3,
    #             alpha=0.3,
    #         )
    #         axins.add_patch(triangle)
    #
    #     # Plot the intersecting triangles
    #     for idx in intersecting_elements:
    #         row = df.iloc[idx]
    #         triangle = Polygon(
    #             [
    #                 (row["element_corner_1_lon"], row["element_corner_1_lat"]),
    #                 (row["element_corner_2_lon"], row["element_corner_2_lat"]),
    #                 (row["element_corner_3_lon"], row["element_corner_3_lat"]),
    #             ],
    #             fill=True,
    #             edgecolor="blue",
    #             facecolor="lightblue",
    #             alpha=0.5,
    #             linewidth=1,
    #         )
    #         axins.add_patch(triangle)
    #
    #     # Plot the transect line
    #     axins.plot(
    #         [actual_first[1], actual_last[1]],
    #         [actual_first[0], actual_last[0]],
    #         "r-",
    #         linewidth=2,
    #     )
    #     axins.plot(start_point[1], start_point[0], "ro", markersize=6)
    #
    #     # Set limits with some padding
    #     all_lons = [row["lon_center"] for _, row in df_transect.iterrows()]
    #     all_lats = [row["lat_center"] for _, row in df_transect.iterrows()]
    #
    #     if len(all_lons) > 0 and len(all_lats) > 0:
    #         lon_range = max(all_lons) - min(all_lons)
    #         lat_range = max(all_lats) - min(all_lats)
    #         padding = 0.5
    #
    #         # Ensure we have non-zero ranges to avoid plotting issues
    #         lon_range = max(0.01, lon_range)
    #         lat_range = max(0.01, lat_range)
    #
    #         axins.set_xlim(
    #             min(all_lons) - padding * lon_range, max(all_lons) + padding * lon_range
    #         )
    #         axins.set_ylim(
    #             min(all_lats) - padding * lat_range, max(all_lats) + padding * lat_range
    #         )
    #
    #     axins.set_title("Transect Location")
    #     axins.set_xlabel("Longitude")
    #     axins.set_ylabel("Latitude")

    return fig, ax


def find_elements_along_line(
    df, kdtree, line, start_point, buffer_dist_deg, km_per_deg_lat, km_per_deg_lon
):
    """
    Helper function to find elements along a line using spatial index
    """
    # Extract line points for querying
    start_coords = list(line.coords)[0]  # (lon, lat)
    end_coords = list(line.coords)[1]

    line_length = np.sqrt(
        (end_coords[0] - start_coords[0]) ** 2 + (end_coords[1] - start_coords[1]) ** 2
    )
    num_points = max(
        100, int(line_length * 100)
    )  # Ensure enough points for accurate searching

    line_points = np.array(
        [
            np.linspace(start_coords[0], end_coords[0], num_points),  # lon
            np.linspace(start_coords[1], end_coords[1], num_points),  # lat
        ]
    ).T

    # Query points near the line
    potential_indices = set()
    for point in line_points:
        indices = kdtree.query_ball_point(point, buffer_dist_deg)
        potential_indices.update(indices)

    print(
        f"Spatial index identified {len(potential_indices)} potential elements near line"
    )

    # Find all triangles that intersect with this line
    intersecting_elements = []
    distances = []

    # Only test elements that are in the potential set
    for idx in potential_indices:
        row = df.iloc[idx]

        # Create polygon from triangle corners
        try:
            triangle = ShapelyPolygon(
                [
                    (row["element_corner_1_lon"], row["element_corner_1_lat"]),
                    (row["element_corner_2_lon"], row["element_corner_2_lat"]),
                    (row["element_corner_3_lon"], row["element_corner_3_lat"]),
                ]
            )

            # Check if the line intersects this triangle
            if triangle.intersects(line):
                # Calculate intersection point(s)
                intersection = triangle.intersection(line)

                # Calculate distance from start point to midpoint of intersection
                if intersection.geom_type == "LineString":
                    midpoint = (
                        (intersection.coords[0][0] + intersection.coords[1][0]) / 2,
                        (intersection.coords[0][1] + intersection.coords[1][1]) / 2,
                    )
                else:  # Point
                    midpoint = intersection.x, intersection.y

                # Convert to km (approximate)
                dist_lon_km = (midpoint[0] - start_point[1]) * km_per_deg_lon
                dist_lat_km = (midpoint[1] - start_point[0]) * km_per_deg_lat
                distance = np.sqrt(dist_lon_km**2 + dist_lat_km**2)

                # Calculate dot product to determine if point is in the desired direction
                vector_to_point = [
                    midpoint[0] - start_point[1],
                    midpoint[1] - start_point[0],
                ]
                direction_vector = [
                    end_coords[0] - start_coords[0],
                    end_coords[1] - start_coords[1],
                ]

                # Normalize direction vector
                dir_magnitude = np.sqrt(
                    direction_vector[0] ** 2 + direction_vector[1] ** 2
                )
                if dir_magnitude > 0:
                    normalized_direction = [
                        direction_vector[0] / dir_magnitude,
                        direction_vector[1] / dir_magnitude,
                    ]
                    dot_product = (
                        vector_to_point[0] * normalized_direction[0]
                        + vector_to_point[1] * normalized_direction[1]
                    )

                    # Only include points in the correct direction
                    if dot_product > 0:
                        intersecting_elements.append(idx)
                        distances.append(distance)
        except Exception as e:
            # Skip any elements with invalid geometry
            print(f"Warning: Skipping element {idx} due to geometry error: {e}")

    # Sort elements by distance along transect
    sorted_pairs = sorted(zip(distances, intersecting_elements))
    sorted_distances = [dist for dist, _ in sorted_pairs]
    sorted_elements = [idx for _, idx in sorted_pairs]

    print(f"Found {len(sorted_elements)} triangular elements intersecting the line")

    return sorted_elements, sorted_distances


def detect_shore(elements, distances):
    """
    Helper function for shore detection - identifies significant gaps that might indicate shore
    """
    if len(elements) <= 1:
        return elements, distances

    # Calculate distances between consecutive elements
    consecutive_dists = []
    for i in range(1, len(distances)):
        consecutive_dists.append(distances[i] - distances[i - 1])

    # Find large gaps that might indicate shore boundaries
    if consecutive_dists:
        avg_dist = np.mean(consecutive_dists)
        std_dist = np.std(consecutive_dists)
        threshold = avg_dist + 5 * std_dist  # Statistical threshold for gaps

        print(f"Average distance between elements: {avg_dist:.2f} km")
        print(f"Standard deviation: {std_dist:.2f} km")
        print(f"Gap threshold for shore detection: {threshold:.2f} km")

        # Find indices where gaps exceed the threshold
        gap_indices = [
            i for i, dist in enumerate(consecutive_dists) if dist > threshold
        ]

        if gap_indices:
            print(
                f"Detected {len(gap_indices)} potential shore boundaries at distances: ",
                end="",
            )
            for idx in gap_indices:
                print(f"{distances[idx]:.2f} km, ", end="")
            print("")

            # If we have gaps, cut the transect to include elements before the first gap
            if len(gap_indices) >= 1:
                first_gap = gap_indices[0]
                elements = elements[: first_gap + 1]
                distances = distances[: first_gap + 1]
                print(
                    f"Trimming transect to first detected shore boundary at {distances[-1]:.2f} km"
                )
        else:
            print("No clear shore boundaries detected based on element gaps")

    return elements, distances


if __name__ == "__main__":
    print("Reading data...")
    ds = xr.open_dataset(
        "../data/WA_puget_sound/b2_summary/001.WA_puget_sound.tidal_hindcast_fvcom-1_year_average.b2.20150101.000000.nc"
    )

    print(ds.info())

    print("Converting to df...")
    df = convert_xarray_to_dataframe(ds)
    # start_point = (48.15, -122.75)  # Near Port Townsend
    # direction_deg = 135  # Southeast direction
    # Selected Admirality Inlet
    start_point = (48.144957, -122.717447)  # Near Port Townsend
    direction_deg = 58
    print("Plotting vert transect")
    fig, axs = plot_vertical_transect(
        df,
        start_point=start_point,
        direction_deg=direction_deg,
        max_distance_km=3,
        variable="speed",
        auto_shore_detect=True,
        bidirectional=True,
    )

    print("Save fig...")

    fig.savefig("transect_test_18.png", dpi=300, bbox_inches="tight")
