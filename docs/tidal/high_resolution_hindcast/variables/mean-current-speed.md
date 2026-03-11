<!-- AUTO-GENERATED FILE — DO NOT EDIT DIRECTLY -->
<!-- Source of truth: src/variable_registry.py (VARIABLE_REGISTRY) -->
<!-- To update: edit the registry, then run `python generate_variable_docs.py` -->

# Mean Current Speed [m/s]

*Annual average of depth-averaged current speed*

## Description

Annual average of the depth-averaged current velocity magnitude, representing the characteristic flow speed at each grid location under free-stream (undisturbed) conditions. This metric is intended for IEC 62600-201 [@iec_62600_201] Stage 1 reconnaissance-level analysis to identify areas with potentially viable tidal current resources. It serves as a primary metric for identifying viable tidal energy sites, used to estimate annual energy production (AEP), compare site potential across regions, determine expected average viable current speeds for commercial deployment, and select appropriate turbine technology.

## Equation

$$
\overline{\overline{U}} = U_{\text{average}} = \text{mean}\left(\left[\text{mean}(U_{1,t}, ..., U_{N_{\sigma},t}) \text{ for } t=1,...,T\right]\right)
$$

**Where:**

- $U_{i,t} = \sqrt{u_{i,t}^2 + v_{i,t}^2}$, velocity magnitude at sigma layer $i$ at time $t$ $[\text{m/s}]$
- $N_{\sigma} = 10$, sigma layers (terrain-following vertical layers dividing the water column into equal-thickness fractions from surface to seafloor)
- $T$, 1 year of hindcast data (hourly for Alaska locations, half-hourly for others)

## Properties

| Property | Value |
| --- | --- |
| Internal Name | `vap_water_column_mean_sea_water_speed` |
| Units | m/s |

--8<-- "docs/tidal/high_resolution_hindcast/_cite-widget.md"
