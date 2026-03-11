<!-- AUTO-GENERATED FILE — DO NOT EDIT DIRECTLY -->
<!-- Source of truth: src/variable_registry.py (VARIABLE_REGISTRY) -->
<!-- To update: edit the registry, then run `python generate_variable_docs.py` -->

# Distance to Shore [NM]

*Geodesic distance from grid cell center to nearest shoreline*

## Description

Distance to Shore is the geodesic distance from each grid cell center to the nearest shoreline point, calculated using the GSHHG (Global Self-consistent Hierarchical High-resolution Geography) shoreline database. Reported in nautical miles (NM). Distance to shore is a practical siting constraint that affects cable cost, grid connection feasibility, and operations and maintenance logistics. NREL site screening methodology uses a threshold of <20 km (~10.8 NM) to nearest transmission infrastructure. Engineering applications include cable routing cost estimation, grid connection planning, and O&M logistics assessment.

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
