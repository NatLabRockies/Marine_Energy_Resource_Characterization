# hotspots module - Part of tidal_visualizer package

import numpy as np
from .utils import haversine_distance


def find_tidal_energy_hotspots(dataset, search_area_m2, min_spacing_m=5000, top_n=5):
    """
    Find the top N most energetic tidal energy hotspots within a dataset using grid search,
    ensuring minimum spacing between hotspots.
    """
    # Calculate search radius in degrees (approximate conversion)
    # The side length of the square in meters
    side_length = np.sqrt(search_area_m2)

    # Convert side length to approximate degrees
    # 1 degree of latitude is approximately 111,000 meters
    search_radius_deg = side_length / 111000

    # Convert minimum spacing to degrees (approximate)
    min_spacing_deg = min_spacing_m / 111000

    # Extract relevant variables for energy assessment
    # We'll use power_density_vert_avg as our primary metric for energy potential
    power_density = dataset.power_density_vert_avg.values
    lats = dataset.lat_center.values
    lons = dataset.lon_center.values

    # Create a structured array with locations and power values
    locations = np.column_stack((lats, lons, power_density))

    # Sort locations by power density (descending)
    sorted_locations = locations[np.argsort(-locations[:, 2])]

    # Initialize list to store hotspots
    hotspots = []

    # Initialize set to track cells that have been included in hotspots
    included_cells = set()

    # Initialize list to track hotspot center coordinates for distance checking
    hotspot_centers = []

    # Process sorted locations to find hotspots
    for i, (lat, lon, power) in enumerate(sorted_locations):
        # Skip if this cell has already been included in a hotspot
        if i in included_cells:
            continue

        # Check if this location is far enough from existing hotspots
        too_close = False
        for center_lat, center_lon in hotspot_centers:
            distance = haversine_distance(lat, lon, center_lat, center_lon)
            if distance < min_spacing_m:
                too_close = True
                break

        if too_close:
            continue

        # Create a new hotspot centered at this location
        hotspot = {
            "center_lat": lat,
            "center_lon": lon,
            "center_power": power,
            "cells": [(i, lat, lon, power)],
            "avg_power": power,
            "total_power": power,
            "min_lat": lat - search_radius_deg / 2,
            "max_lat": lat + search_radius_deg / 2,
            "min_lon": lon - search_radius_deg / 2,
            "max_lon": lon + search_radius_deg / 2,
        }

        # Find all cells within the search area
        for j, (j_lat, j_lon, j_power) in enumerate(sorted_locations):
            if j == i or j in included_cells:
                continue

            # Check if this cell is within the search area
            if (
                abs(j_lat - lat) <= search_radius_deg / 2
                and abs(j_lon - lon) <= search_radius_deg / 2
            ):
                # Add to this hotspot
                hotspot["cells"].append((j, j_lat, j_lon, j_power))
                hotspot["total_power"] += j_power

                # Mark as included
                included_cells.add(j)

        # Update average power
        hotspot["avg_power"] = hotspot["total_power"] / len(hotspot["cells"])

        # Add this hotspot to our list
        hotspots.append(hotspot)

        # Add center to our list of centers
        hotspot_centers.append((lat, lon))

        # Stop once we have top_n hotspots
        if len(hotspots) >= top_n:
            break

    # If we didn't find enough hotspots yet,
    # gradually reduce the minimum spacing requirement
    original_min_spacing = min_spacing_m
    reduction_factor = 0.8  # Reduce by 20% each iteration

    while len(hotspots) < top_n and min_spacing_m > original_min_spacing * 0.3:
        # Reduce minimum spacing
        min_spacing_m *= reduction_factor
        print(
            f"Reducing minimum spacing to {min_spacing_m:.1f} meters to find more hotspots"
        )

        for i, (lat, lon, power) in enumerate(sorted_locations):
            if i in included_cells:
                continue

            # Check if this location is far enough from existing hotspots
            too_close = False
            for center_lat, center_lon in hotspot_centers:
                distance = haversine_distance(lat, lon, center_lat, center_lon)
                if distance < min_spacing_m:
                    too_close = True
                    break

            if too_close:
                continue

            # Create a new hotspot
            hotspot = {
                "center_lat": lat,
                "center_lon": lon,
                "center_power": power,
                "cells": [(i, lat, lon, power)],
                "avg_power": power,
                "total_power": power,
                "min_lat": lat - search_radius_deg / 2,
                "max_lat": lat + search_radius_deg / 2,
                "min_lon": lon - search_radius_deg / 2,
                "max_lon": lon + search_radius_deg / 2,
            }

            # Find all cells within the search area
            for j, (j_lat, j_lon, j_power) in enumerate(sorted_locations):
                if j == i or j in included_cells:
                    continue

                # Check if this cell is within the search area
                if (
                    abs(j_lat - lat) <= search_radius_deg / 2
                    and abs(j_lon - lon) <= search_radius_deg / 2
                ):
                    # Add to this hotspot
                    hotspot["cells"].append((j, j_lat, j_lon, j_power))
                    hotspot["total_power"] += j_power

                    # Mark as included
                    included_cells.add(j)

            # Update average power
            hotspot["avg_power"] = hotspot["total_power"] / len(hotspot["cells"])

            # Add this hotspot to our list
            hotspots.append(hotspot)

            # Add center to our list of centers
            hotspot_centers.append((lat, lon))

            if len(hotspots) >= top_n:
                break

    # Format the results
    results = []
    for i, hotspot in enumerate(hotspots):
        # Calculate distances to other hotspots
        distances_to_others = []
        for j, other_hotspot in enumerate(hotspots):
            if i != j:
                dist = haversine_distance(
                    hotspot["center_lat"],
                    hotspot["center_lon"],
                    other_hotspot["center_lat"],
                    other_hotspot["center_lon"],
                )
                distances_to_others.append(dist)

        # Calculate minimum distance to another hotspot
        min_distance = min(distances_to_others) if distances_to_others else float("inf")

        results.append(
            {
                "rank": i + 1,
                "center_latitude": hotspot["center_lat"],
                "center_longitude": hotspot["center_lon"],
                "average_power_density": hotspot["avg_power"],
                "peak_power_density": hotspot["center_power"],
                "num_cells": len(hotspot["cells"]),
                "min_distance_to_other_hotspot_m": min_distance,
                "search_area_bounds": {
                    "min_lat": hotspot["min_lat"],
                    "max_lat": hotspot["max_lat"],
                    "min_lon": hotspot["min_lon"],
                    "max_lon": hotspot["max_lon"],
                },
            }
        )

    return results
