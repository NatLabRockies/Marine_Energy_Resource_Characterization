"""
GIS color styling registry for tidal hindcast variables.

Defines colormap names (as strings, no matplotlib/cmocean import at module scope),
numeric ranges, and levels for each styled variable. Keyed by the same variable keys
as VARIABLE_REGISTRY.
"""


GIS_COLORS_REGISTRY = {
    # --- Continuous colormaps ---
    "mean_current_speed": {
        "style_type": "continuous",
        "colormap_name": "cmo.thermal",
        "range_min": 0,
        "range_max": 1.5,
        "levels": 10,
    },
    "p95_current_speed": {
        "style_type": "continuous",
        "colormap_name": "cmo.matter",
        "range_min": 0,
        "range_max": 5.0,
        "levels": 10,
    },
    "mean_power_density": {
        "style_type": "continuous",
        "colormap_name": "cmo.dense",
        "range_min": 0,
        "range_max": 1750,
        "levels": 7,
    },
    "min_water_depth": {
        "style_type": "continuous",
        "colormap_name": "cmo.deep",
        "range_min": 0,
        "range_max": 200,
        "levels": 10,
    },
    "max_water_depth": {
        "style_type": "continuous",
        "colormap_name": "cmo.deep",
        "range_min": 0,
        "range_max": 200,
        "levels": 10,
    },
    # --- Discrete / categorical ---
    "grid_resolution": {
        "style_type": "discrete",
        "range_min": 0,
        "range_max": 500,
        "levels": 3,
        "spec_ranges": {
            "stage_2": {
                "max": 50,
                "label": "Stage 2 (≤50m)",
                "color": "#1f77b4",
            },
            "stage_1": {
                "max": 500,
                "label": "Stage 1 (≤500m)",
                "color": "#ff7f0e",
            },
            "non_compliant": {
                "max": 100000,
                "label": "Non-compliant (>500m)",
                "color": "#DC143C",
            },
        },
    },
}


def resolve_colormap(name):
    """Resolve a colormap name string to a matplotlib colormap object.

    Importing cmocean registers its colormaps with matplotlib under the
    ``cmo.`` prefix (e.g. ``"cmo.thermal"``).  Any matplotlib builtin
    name (e.g. ``"viridis"``) also works.
    """
    import cmocean  # noqa: F401 -- registers cmo.* colormaps with matplotlib
    import matplotlib as mpl

    return mpl.colormaps[name]