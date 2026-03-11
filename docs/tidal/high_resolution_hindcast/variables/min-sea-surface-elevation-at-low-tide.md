<!-- AUTO-GENERATED FILE — DO NOT EDIT DIRECTLY -->
<!-- Source of truth: src/variable_registry.py (VARIABLE_REGISTRY) -->
<!-- To update: edit the registry, then run `python generate_variable_docs.py` -->

# Minimum Sea Surface Elevation at Low Tide [m (relative to model MSL)]

*Lowest sea surface elevation observed during low tide over the hindcast year*

## Description

Minimum Sea Surface Elevation at Low Tide is the lowest sea surface elevation value observed during low tide conditions over the hindcast year, relative to the model's mean sea level. This typically occurs during spring tides when astronomical tidal forcing is maximized. This metric serves as a sanity check for data quality, allowing coastal engineers and oceanographers to verify that modeled extreme low water levels are physically reasonable for the region. Together with the high tide maximum, it provides confidence bounds on the vertical water level envelope.

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
