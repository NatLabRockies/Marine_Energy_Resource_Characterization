<!-- AUTO-GENERATED FILE — DO NOT EDIT DIRECTLY -->
<!-- Source of truth: src/variable_registry.py (VARIABLE_REGISTRY) -->
<!-- To update: edit the registry, then run `python generate_variable_docs.py` -->

# Minimum Water Depth [m]

*Minimum water depth calculated over the 1-year hindcast period*

## Description

The minimum water depth (surface to seafloor) calculated at each grid location over the 1-year hindcast period, typically occurring during extreme low tide conditions. This metric represents the model lower bound of water depth variability and is intended for assessing deployment feasibility, defining safe navigation limits, and planning installation operations.

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
