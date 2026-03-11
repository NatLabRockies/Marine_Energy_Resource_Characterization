<!-- AUTO-GENERATED FILE — DO NOT EDIT DIRECTLY -->
<!-- Source of truth: src/variable_registry.py (VARIABLE_REGISTRY) -->
<!-- To update: edit the registry, then run `python generate_variable_docs.py` -->

# Maximum Water Depth [m]

*Maximum water depth (during 1 year model run)*

## Description

Maximum Water Depth is the greatest water depth (surface to seafloor) observed at each grid location over the hindcast year, typically occurring during extreme high tide conditions. This metric represents the upper bound of water depth variability at each location. Used in Stage 2 feasibility studies for mooring system design, cable routing, and understanding the full operating depth envelope. The difference between maximum and minimum water depth approximates the tidal range at each location. Engineering applications include mooring system design considerations and understanding the full range of water depths at a site.

## Equation

$$
d_{\max} = \max\left(\left[(h + \zeta_t) \text{ for } t=1,...,T\right]\right)
$$

**Where:**

- $h$, bathymetry below NAVD88 $[\text{m}]$
- $\zeta_t$, sea surface elevation above NAVD88 at time $t$ $[\text{m}]$
- $T$, 1 year of hindcast data (hourly for Alaska locations, half-hourly for others)

## Properties

| Property | Value |
| --- | --- |
| Internal Name | `vap_water_column_height_max` |
| Units | m |

--8<-- "docs/tidal/high_resolution_hindcast/_cite-widget.md"
