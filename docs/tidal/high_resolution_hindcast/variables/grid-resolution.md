<!-- AUTO-GENERATED FILE — DO NOT EDIT DIRECTLY -->
<!-- Source of truth: src/variable_registry.py (VARIABLE_REGISTRY) -->
<!-- To update: edit the registry, then run `python generate_variable_docs.py` -->

# Grid Resolution [m]

*Average edge length of triangular model grid cells*

## Description

Average edge length of the unstructured triangular model grid cells, indicating the spatial scale at which tidal currents are resolved by the FVCOM hydrodynamic model. This metric is essential model metadata for assessing spatial uncertainty and determining appropriate applications. The unstructured triangular mesh allows variable resolution, with finer grids in areas of interest (channels, straits) and coarser grids in open water. According to IEC 62600-201 standards [@iec_62600_201], Stage 1 reconnaissance-level assessments require grid resolution < 500 m, while Stage 2 layout design assessments require grid resolution < 50 m.

## Equation

$$
\text{Grid Resolution} = \frac{1}{3}(d_1 + d_2 + d_3)
$$

**Where:**

- $d_1, d_2, d_3$, geodesic distances between triangle vertices $[\text{m}]$

## Properties

| Property | Value |
| --- | --- |
| Internal Name | `vap_grid_resolution` |
| Units | m |

--8<-- "docs/tidal/high_resolution_hindcast/_cite-widget.md"
