<!-- AUTO-GENERATED FILE — DO NOT EDIT DIRECTLY -->
<!-- Source of truth: src/variable_registry.py (VARIABLE_REGISTRY) -->
<!-- To update: edit the registry, then run `python generate_variable_docs.py` -->

# Maximum Sea Surface Elevation at High Tide [m (relative to model MSL)]

*Highest sea surface elevation observed during high tide over the hindcast year*

## Description

Maximum Sea Surface Elevation at High Tide is the highest sea surface elevation value observed during high tide conditions over the hindcast year, relative to the model's mean sea level. This typically occurs during spring tides when astronomical tidal forcing is maximized. This metric serves as a sanity check for data quality, allowing coastal engineers and oceanographers to verify that modeled extreme water levels are physically reasonable for the region. Together with the low tide minimum, it provides confidence bounds on the vertical water level envelope.

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
