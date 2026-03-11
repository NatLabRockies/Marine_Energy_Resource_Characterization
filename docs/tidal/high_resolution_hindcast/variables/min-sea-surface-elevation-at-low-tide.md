<!-- AUTO-GENERATED FILE — DO NOT EDIT DIRECTLY -->
<!-- Source of truth: src/variable_registry.py (VARIABLE_REGISTRY) -->
<!-- To update: edit the registry, then run `python generate_variable_docs.py` -->

# Minimum Sea Surface Elevation at Low Tide [m (relative to model MSL)]

*Lowest sea surface elevation calculated during low tide over the hindcast year*

## Description

Lowest sea surface elevation calculated during low tide conditions over the 1-year hindcast period, relative to the model's mean sea level. This typically occurs during spring tides when astronomical tidal forcing is maximized. Together with the high tide maximum, it defines the bounds of the vertical water level envelope. Note that this value represents the minimum within the 1-year hindcast period, not necessarily the absolute minimum possible over the full 18.6-year tidal epoch.

## Equation

$$
\zeta_{\text{LT,min}} = \min(\zeta_t | t \in \text{low tide troughs})
$$

**Where:**

- $\zeta_t$, sea surface elevation relative to model MSL at time $t$ $[\text{m}]$
- Low tide troughs identified from the sea surface elevation time series

## Properties

| Property | Value |
| --- | --- |
| Internal Name | `vap_surface_elevation_low_tide_min` |
| Units | m (relative to model MSL) |

--8<-- "docs/tidal/high_resolution_hindcast/_cite-widget.md"
