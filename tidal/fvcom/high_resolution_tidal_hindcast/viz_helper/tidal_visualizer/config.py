# config module - Part of tidal_visualizer package


def get_location_names(config):
    """Return a list of all location keys in the configuration."""
    locations_section = config.get("locations", {})
    return list(locations_section.keys())


def get_areas_of_interest(config, location_key):
    """Return a list of all area of interest keys for a given location."""
    try:
        return list(config["locations"][location_key]["hotspots"].keys())
    except (KeyError, TypeError):
        return []


def get_location_info(config, location_key):
    """Return the configuration information for a specific location."""
    try:
        return config["locations"][location_key]
    except (KeyError, TypeError):
        return None


def get_area_of_interest_info(config, location_key, area_key):
    """Return the configuration information for a specific area of interest."""
    try:
        return config["locations"][location_key]["hotspots"][area_key]
    except (KeyError, TypeError):
        return None


def get_variable_range(config, variable_name):
    """Return the min and max range for a variable if defined in config."""
    try:
        variable_config = config["variables"][variable_name]
        return variable_config.get("min"), variable_config.get("max")
    except (KeyError, TypeError):
        return None, None


def add_hotspots_to_config(
    config, location_key, hotspots, search_area_m2, prefix="auto_"
):
    """Add automatically detected hotspots to the configuration."""
    if location_key not in config.get("locations", {}):
        return config

    for hotspot in hotspots:
        hotspot_key = f"{prefix}hotspot_{hotspot['rank']}"
        bounds = hotspot["search_area_bounds"]

        # Skip if the hotspot key already exists
        if hotspot_key in config["locations"][location_key]["hotspots"]:
            continue

        # Create the hotspot entry
        config["locations"][location_key]["hotspots"][hotspot_key] = {
            "display_name": f"Hotspot {hotspot['rank']} ({hotspot['peak_power_density']:.1f} W/m²)",
            "bounds": [
                bounds["min_lon"],
                bounds["min_lat"],
                bounds["max_lon"],
                bounds["max_lat"],
            ],
            "peak_power_density": hotspot["peak_power_density"],
            "average_power_density": hotspot["average_power_density"],
        }

    return config
