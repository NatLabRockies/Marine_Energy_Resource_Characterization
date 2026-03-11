<!-- AUTO-GENERATED FILE — DO NOT EDIT DIRECTLY -->
<!-- Source of truth: src/variable_registry.py (VARIABLE_REGISTRY) -->
<!-- To update: edit the registry, then run `python generate_variable_docs.py` -->

# Minimum Water Depth [m]

*Minimum water depth (during 1 year model run)*

## Description

Minimum Water Depth is the lowest water depth (surface to seafloor) observed at each grid location over the hindcast year, typically occurring during extreme low tide conditions. This metric defines the minimum vertical clearance available for device deployment and is critical for assessing depth constraints. Used in Stage 2 feasibility studies for turbine placement and collision avoidance. The difference between maximum and minimum water depth approximates the tidal range at each location. Engineering applications include assessing turbine clearance requirements and identifying areas where shallow water may limit device deployment.

## Equation

$$
d_{\min} = \min\left(\left[(h + \zeta_t) \text{ for } t=1,...,T\right]\right)
$$

**Where:**

- $h$, bathymetry below NAVD88 $[\text{m}]$
- $\zeta_t$, sea surface elevation above NAVD88 at time $t$ $[\text{m}]$
- $T$, 1 year of hindcast data (hourly for Alaska locations, half-hourly for others)

## Properties

| Property | Value |
| --- | --- |
| Internal Name | `vap_water_column_height_min` |
| Units | m |

--8<-- "docs/tidal/high_resolution_hindcast/_cite-widget.md"
