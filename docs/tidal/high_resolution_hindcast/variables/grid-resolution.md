<!-- AUTO-GENERATED FILE — DO NOT EDIT DIRECTLY -->
<!-- Source of truth: src/variable_registry.py (VARIABLE_REGISTRY) -->
<!-- To update: edit the registry, then run `python generate_variable_docs.py` -->

# Grid Resolution [m]

*Average edge length of triangular model grid cells*

## Description

Grid Resolution is the average edge length of the unstructured triangular model grid cells, indicating the spatial scale at which tidal currents are resolved by the FVCOM hydrodynamic model. Essential model metadata for assessing spatial uncertainty and determining appropriate applications. IEC 62600-201 requires <500 m for Stage 1 reconnaissance and <50 m for Stage 2 feasibility assessments. The unstructured triangular mesh allows variable resolution, with finer grids in areas of interest (channels, straits) and coarser grids in open water. Per IEC 62600-201 tidal energy resource assessment standards:- Stage 1 feasibility (reconnaissance-level) assessments require grid resolution < 500 m- Stage 2 (layout design) assessments require grid resolution < 50 mEngineering applications include assessing model fidelity and determining appropriate applications for the data.

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
