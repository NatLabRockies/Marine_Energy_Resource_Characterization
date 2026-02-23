"""
GIS color styling manager for tidal hindcast outputs.

Public API:
  - add_fill_color_columns(gdf)       : ``fill_color_<col>`` hex columns
  - embed_gpkg_styles(gpkg_path)      : OGC SLD styles in GPKG layer_styles table
  - write_sidecar_style_json(path, gdf) : generic style spec JSON
  - build_kepler_config(gdf)          : kepler.gl-compatible config dict
  - write_kepler_config_json(path, gdf) : kepler_config.json sidecar
  - embed_kepler_config_in_parquet(path, gdf) : append kepler config to parquet metadata
"""

import json
import sqlite3
from pathlib import Path

import matplotlib.colors as mcolors
import numpy as np
from lxml import etree

from .gis_colors_registry import GIS_COLORS_REGISTRY, resolve_colormap
from .variable_registry import VARIABLE_REGISTRY


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def add_fill_color_columns(gdf):
    """Add ``fill_color_{column_name}`` hex-color columns for each styled variable.

    Intended for GeoJSON and GeoParquet outputs (not GPKG, which uses SLD styles).

    Returns
    -------
    geopandas.GeoDataFrame
        Same frame with additional ``fill_color_`` columns.
    """
    for var_key, style in GIS_COLORS_REGISTRY.items():
        reg = VARIABLE_REGISTRY.get(var_key)
        if reg is None:
            continue
        col = reg["column_name"]
        if col not in gdf.columns:
            continue

        fill_col = f"fill_color_{col}"

        if style["style_type"] == "continuous":
            gdf[fill_col] = _continuous_hex_colors(
                gdf[col].values,
                style["colormap_name"],
                style["range_min"],
                style["range_max"],
                style["levels"],
            )
        elif style["style_type"] == "discrete":
            gdf[fill_col] = _discrete_hex_colors(
                gdf[col].values,
                style["spec_ranges"],
            )

    return gdf


def embed_gpkg_styles(gpkg_path):
    """Write OGC SLD style definitions into the GPKG ``layer_styles`` table.

    Uses `python-sld <https://github.com/azavea/python-sld>`_ to build
    proper graduated (continuous) and categorized (discrete) styles that
    reference the **numeric data column** directly.

    QGIS reads this table automatically on layer load.
    """
    gpkg_path = str(gpkg_path)

    conn = sqlite3.connect(gpkg_path)
    try:
        cur = conn.cursor()
        geom_col = _get_gpkg_geometry_column(cur)
        layer_name = _get_gpkg_layer_name(cur)
        cur.execute(_LAYER_STYLES_DDL)

        is_first = True
        for var_key, style in GIS_COLORS_REGISTRY.items():
            reg = VARIABLE_REGISTRY.get(var_key)
            if reg is None:
                continue
            col = reg["column_name"]
            display = reg.get("display_name", var_key)

            sld_xml = _build_sld(layer_name, col, display, style)

            cur.execute(
                """
                INSERT INTO layer_styles
                    (f_table_catalog, f_table_schema, f_table_name,
                     f_geometry_column, styleName, styleQML, styleSLD,
                     useAsDefault, description, owner, ui, update_time)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
                """,
                (
                    "",
                    "",
                    layer_name,
                    geom_col,
                    display,
                    "",
                    sld_xml,
                    1 if is_first else 0,
                    f"Auto-generated style for {display}",
                    "gis_colors_manager",
                    "",
                ),
            )
            is_first = False

        conn.commit()
    finally:
        conn.close()


def write_sidecar_style_json(output_path, gdf):
    """Write ``gis_style_spec.json`` alongside GIS outputs."""
    output_path = Path(output_path)
    spec = {
        "schema_version": "1.0",
        "description": "GIS color styling specification for tidal hindcast variables",
        "variables": {},
    }

    for var_key, style in GIS_COLORS_REGISTRY.items():
        reg = VARIABLE_REGISTRY.get(var_key)
        if reg is None:
            continue
        col = reg["column_name"]
        fill_col = f"fill_color_{col}"
        if fill_col not in gdf.columns:
            continue

        entry = {
            "column_name": col,
            "fill_color_column": fill_col,
            "display_name": reg.get("display_name", var_key),
            "units": reg.get("units", ""),
            "style_type": style["style_type"],
        }

        if style["style_type"] == "continuous":
            entry["colormap_name"] = style["colormap_name"]
            entry["range_min"] = style["range_min"]
            entry["range_max"] = style["range_max"]
            entry["levels"] = style["levels"]
            entry["color_levels"] = _build_color_level_list(
                style["colormap_name"],
                style["range_min"],
                style["range_max"],
                style["levels"],
            )
        elif style["style_type"] == "discrete":
            entry["categories"] = {
                k: {"max": v["max"], "label": v["label"], "color": v["color"]}
                for k, v in style["spec_ranges"].items()
            }

        spec["variables"][var_key] = entry

    json_path = output_path / "gis_style_spec.json"
    with open(json_path, "w") as f:
        json.dump(spec, f, indent=2)
    print(f"  Wrote sidecar style spec: {json_path}")


def build_kepler_config(gdf):
    """Build a kepler.gl-compatible map config dict.

    Generates one geojson layer per styled variable, each with a
    ``colorField`` referencing the numeric data column and a
    ``colorRange`` with the matching hex palette.
    """
    layers = []
    for var_key, style in GIS_COLORS_REGISTRY.items():
        reg = VARIABLE_REGISTRY.get(var_key)
        if reg is None:
            continue
        col = reg["column_name"]
        if col not in gdf.columns:
            continue
        display = reg.get("display_name", var_key)

        if style["style_type"] == "continuous":
            _, hex_palette = _make_hex_palette(
                style["colormap_name"],
                style["range_min"],
                style["range_max"],
                style["levels"],
            )
            color_range = {
                "name": style["colormap_name"],
                "type": "custom",
                "category": "Custom",
                "colors": hex_palette,
            }
            color_scale = "quantize"
            color_field_type = "real"
        elif style["style_type"] == "discrete":
            sorted_specs = sorted(
                style["spec_ranges"].values(), key=lambda s: s["max"]
            )
            color_range = {
                "name": f"{var_key}_categories",
                "type": "custom",
                "category": "Custom",
                "colors": [s["color"] for s in sorted_specs],
            }
            color_scale = "quantize"
            color_field_type = "real"
        else:
            continue

        layer = {
            "id": var_key,
            "type": "geojson",
            "config": {
                "dataId": "tidal_hindcast",
                "label": display,
                "isVisible": var_key == "mean_current_speed",
                "visConfig": {
                    "opacity": 0.85,
                    "filled": True,
                    "stroked": False,
                    "colorRange": color_range,
                },
            },
            "visualChannels": {
                "colorField": {"name": col, "type": color_field_type},
                "colorScale": color_scale,
            },
        }
        layers.append(layer)

    return {
        "version": "v1",
        "config": {
            "visState": {
                "layers": layers,
            },
        },
    }


def write_kepler_config_json(output_path, gdf):
    """Write ``kepler_config.json`` sidecar file."""
    output_path = Path(output_path)
    config = build_kepler_config(gdf)
    json_path = output_path / "kepler_config.json"
    with open(json_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Wrote kepler config: {json_path}")


def embed_kepler_config_in_parquet(parquet_path, gdf):
    """Append kepler.gl config to existing Parquet file metadata.

    Reads the file, merges the kepler config into the schema metadata
    (preserving all existing keys like ``geo``), and rewrites.
    """
    import pyarrow.parquet as pq

    parquet_path = str(parquet_path)
    config = build_kepler_config(gdf)

    # Read existing file and metadata
    table = pq.read_table(parquet_path)
    existing_meta = table.schema.metadata or {}

    # Append kepler config (don't overwrite existing keys)
    new_meta = {**existing_meta, b"kepler_config": json.dumps(config).encode("utf-8")}
    table = table.replace_schema_metadata(new_meta)

    pq.write_table(table, parquet_path)
    print(f"  Embedded kepler config in parquet metadata: {parquet_path}")


# ---------------------------------------------------------------------------
# Internal helpers — color computation
# ---------------------------------------------------------------------------


def _make_norm_and_cmap(colormap_name, vmin, vmax, levels):
    """Build a BoundaryNorm + colormap pair and return (edges, norm, cmap).

    Uses matplotlib's ``BoundaryNorm`` for discretised mapping so that
    the full Normalize → Colormap → to_hex pipeline is handled by
    matplotlib rather than manual bin arithmetic.
    """
    cmap = resolve_colormap(colormap_name)
    edges = np.linspace(vmin, vmax, levels + 1)
    norm = mcolors.BoundaryNorm(edges, cmap.N, clip=True)
    return edges, norm, cmap


def _make_hex_palette(colormap_name, vmin, vmax, levels):
    """Build bin edges and a hex color palette for a discretised colormap.

    Samples the colormap at bin centres via the norm so colours are
    consistent with ``_continuous_hex_colors``.
    """
    edges, norm, cmap = _make_norm_and_cmap(colormap_name, vmin, vmax, levels)
    bin_centers = (edges[:-1] + edges[1:]) / 2.0
    hex_palette = [mcolors.to_hex(cmap(norm(v))) for v in bin_centers]
    return edges, hex_palette


def _continuous_hex_colors(values, colormap_name, vmin, vmax, levels):
    """Map numeric *values* to hex color strings via matplotlib's norm → cmap pipeline."""
    edges, norm, cmap = _make_norm_and_cmap(colormap_name, vmin, vmax, levels)
    values = np.asarray(values, dtype=float)

    # Normalize → colormap → hex (matplotlib handles clipping)
    normed = norm(np.nan_to_num(values, nan=vmin))
    rgba = cmap(normed)
    hex_colors = np.array([mcolors.to_hex(c) for c in rgba], dtype=object)

    # NaN → black
    hex_colors[np.isnan(values)] = "#000000"
    return hex_colors.tolist()


def _discrete_hex_colors(values, spec_ranges):
    """Map numeric *values* to hex colors using a ListedColormap + BoundaryNorm."""
    sorted_specs = sorted(spec_ranges.values(), key=lambda s: s["max"])
    colors = [s["color"] for s in sorted_specs]
    boundaries = [0] + [s["max"] for s in sorted_specs]

    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(boundaries, cmap.N, clip=True)

    values = np.asarray(values, dtype=float)
    rgba = cmap(norm(np.nan_to_num(values, nan=0)))
    hex_colors = np.array([mcolors.to_hex(c) for c in rgba], dtype=object)
    hex_colors[np.isnan(values)] = "#000000"
    return hex_colors.tolist()


def _build_color_level_list(colormap_name, vmin, vmax, levels):
    """Return a list of dicts describing each color level (for JSON spec)."""
    edges, hex_palette = _make_hex_palette(colormap_name, vmin, vmax, levels)
    return [
        {"bin_min": float(edges[i]), "bin_max": float(edges[i + 1]), "color": hex_palette[i]}
        for i in range(levels)
    ]


# ---------------------------------------------------------------------------
# GPKG / SLD helpers (using python-sld)
# ---------------------------------------------------------------------------

_LAYER_STYLES_DDL = """
CREATE TABLE IF NOT EXISTS layer_styles (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    f_table_catalog  TEXT DEFAULT '',
    f_table_schema   TEXT DEFAULT '',
    f_table_name     TEXT NOT NULL,
    f_geometry_column TEXT NOT NULL,
    styleName    TEXT NOT NULL,
    styleQML     TEXT DEFAULT '',
    styleSLD     TEXT DEFAULT '',
    useAsDefault INTEGER DEFAULT 0,
    description  TEXT DEFAULT '',
    owner        TEXT DEFAULT '',
    ui           TEXT DEFAULT '',
    update_time  TEXT DEFAULT (datetime('now'))
)
"""


def _get_gpkg_geometry_column(cursor):
    """Query gpkg_geometry_columns for the geometry column name."""
    try:
        cursor.execute("SELECT column_name FROM gpkg_geometry_columns LIMIT 1")
        row = cursor.fetchone()
        if row:
            return row[0]
    except sqlite3.OperationalError:
        pass
    return "geometry"


def _get_gpkg_layer_name(cursor):
    """Query gpkg_contents for the layer/table name."""
    try:
        cursor.execute("SELECT table_name FROM gpkg_contents LIMIT 1")
        row = cursor.fetchone()
        if row:
            return row[0]
    except sqlite3.OperationalError:
        pass
    return "data"


def _build_sld(layer_name, data_col, display_name, style):
    """Build an SLD XML string using python-sld.

    Continuous variables get graduated rules with ``PropertyIsBetween``-style
    AND filters on the numeric column.  Discrete variables get categorized
    rules with threshold filters.
    """
    import sld as pysld

    s = pysld.StyledLayerDescriptor()
    nl = s.create_namedlayer(layer_name)
    us = nl.create_userstyle()
    us.Title = display_name
    fts = us.create_featuretypestyle()

    if style["style_type"] == "continuous":
        _add_graduated_rules(fts, data_col, style)
    elif style["style_type"] == "discrete":
        _add_categorized_rules(fts, data_col, style)

    # Pretty-print the SLD XML
    xml_bytes = s.as_sld()
    tree = etree.fromstring(xml_bytes)
    return etree.tostring(tree, pretty_print=True, xml_declaration=True, encoding="UTF-8").decode()


def _configure_polygon_symbolizer(rule, fill_color):
    """Set fill color and opacity, remove stroke on a polygon rule."""
    for cp in rule.PolygonSymbolizer.Fill.CssParameters:
        if cp.Name == "fill":
            cp.Value = fill_color
    rule.PolygonSymbolizer.Fill.create_cssparameter("fill-opacity", "0.85")
    for cp in rule.PolygonSymbolizer.Stroke.CssParameters:
        if cp.Name == "stroke-width":
            cp.Value = "0"


def _create_and_filter(fts, rule, data_col, op1, val1, op2, val2):
    """Create an AND filter (range) on *rule* using a temporary rule for the second condition."""
    f1 = rule.create_filter(data_col, op1, str(val1))
    temp = fts.create_rule("_temp", pysld.PolygonSymbolizer)
    f2 = temp.create_filter(data_col, op2, str(val2))
    rule.Filter = f1 + f2
    temp._node.getparent().remove(temp._node)


def _add_graduated_rules(fts, data_col, style):
    """Add graduated (continuous) rules to *fts*."""
    import sld as pysld

    edges, hex_palette = _make_hex_palette(
        style["colormap_name"], style["range_min"], style["range_max"], style["levels"],
    )

    for i in range(style["levels"]):
        lo, hi, color = float(edges[i]), float(edges[i + 1]), hex_palette[i]
        title = f"{lo:.4g} - {hi:.4g}"

        rule = fts.create_rule(title, pysld.PolygonSymbolizer)
        _configure_polygon_symbolizer(rule, color)

        # AND filter: >= lo AND < hi (last bin uses <=)
        f1 = rule.create_filter(data_col, ">=", str(lo))
        temp = fts.create_rule("_temp", pysld.PolygonSymbolizer)
        op2 = "<=" if i == style["levels"] - 1 else "<"
        f2 = temp.create_filter(data_col, op2, str(hi))
        rule.Filter = f1 + f2
        temp._node.getparent().remove(temp._node)


def _add_categorized_rules(fts, data_col, style):
    """Add categorized (discrete) rules to *fts*."""
    import sld as pysld

    sorted_specs = sorted(style["spec_ranges"].items(), key=lambda kv: kv[1]["max"])
    prev_max = None

    for _, spec in sorted_specs:
        rule = fts.create_rule(spec["label"], pysld.PolygonSymbolizer)
        _configure_polygon_symbolizer(rule, spec["color"])

        if prev_max is None:
            # First category: just <= max
            rule.create_filter(data_col, "<=", str(spec["max"]))
        else:
            # Subsequent: > prev_max AND <= max
            f1 = rule.create_filter(data_col, ">", str(prev_max))
            temp = fts.create_rule("_temp", pysld.PolygonSymbolizer)
            f2 = temp.create_filter(data_col, "<=", str(spec["max"]))
            rule.Filter = f1 + f2
            temp._node.getparent().remove(temp._node)

        prev_max = spec["max"]
