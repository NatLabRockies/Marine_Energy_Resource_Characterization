<!-- AUTO-GENERATED FILE — DO NOT EDIT DIRECTLY -->
<!-- Source of truth: src/variable_registry.py (VARIABLE_REGISTRY) -->
<!-- To update: edit the registry, then run `python generate_variable_docs.py` -->

# Distance to Shore [NM]

*Geodesic distance from grid cell center to nearest shoreline*

## Description

Geodesic distance from each grid cell center to the nearest shoreline point, calculated using the Global Self-consistent Hierarchical High-resolution Geography (GSHHG) shoreline database [@gsshhg_dataset] and reported in nautical miles (NM). This metric serves as a practical siting constraint that affects cable cost and grid connection feasibility.

## Equation

$$
d = \text{haversine}(\text{cell center}, \text{nearest shoreline point})
$$

**Where:**

- $d$, geodesic distance calculated using GSHHG shoreline database $[\text{NM}]$
- $1 \text{ NM} = 1.852 \text{ km}$

## Properties

| Property | Value |
| --- | --- |
| Internal Name | `vap_distance_to_shore` |
| Units | NM |

--8<-- "docs/tidal/high_resolution_hindcast/_cite-widget.md"
