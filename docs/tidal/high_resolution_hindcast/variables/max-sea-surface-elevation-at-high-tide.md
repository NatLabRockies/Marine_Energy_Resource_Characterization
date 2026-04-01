<!-- AUTO-GENERATED FILE — DO NOT EDIT DIRECTLY -->
<!-- Source of truth: src/variable_registry.py (VARIABLE_REGISTRY) -->
<!-- To update: edit the registry, then run `python generate_variable_docs.py` -->

# Maximum Sea Surface Elevation at High Tide [m (relative to model MSL)]

*Highest sea surface elevation calculated during high tide over the hindcast year*

## Description

Highest sea surface elevation calculated during high tide conditions over the 1-year hindcast period, relative to the model's mean sea level. This typically occurs during spring tides when astronomical tidal forcing is maximized. Note that this value represents the maximum within the 1-year hindcast period, not necessarily the absolute maximum possible over the full 18.6-year tidal epoch (e.g., King Tides).

## Equation

$$
\zeta_{\text{HT,max}} = \max(\zeta_t | t \in \text{high tide peaks})
$$

**Where:**

- $\zeta_t$, sea surface elevation relative to model MSL at time $t$ $[\text{m}]$
- High tide peaks identified from the sea surface elevation time series

## Properties

| Property | Value |
| --- | --- |
| Internal Name | `vap_sea_surface_elevation_high_tide_max` |
| Units | m (relative to model MSL) |

--8<-- "docs/tidal/high_resolution_hindcast/_cite-widget.md"
