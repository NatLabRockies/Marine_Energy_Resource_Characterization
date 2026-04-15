<!-- AUTO-GENERATED FILE — DO NOT EDIT DIRECTLY -->
<!-- Source of truth: src/variable_registry.py (VARIABLE_REGISTRY) -->
<!-- To update: edit the registry, then run `python generate_variable_docs.py` -->

# Maximum Water Depth [m]

*Maximum water depth calculated over the 1-year hindcast period*

## Description

Maximum water depth (surface to seafloor) calculated at each grid location over the 1-year hindcast period, typically occurring during extreme high tide conditions. This metric represents the upper bound of water depth variability at each location and defines the maximum expected water column depth.

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
