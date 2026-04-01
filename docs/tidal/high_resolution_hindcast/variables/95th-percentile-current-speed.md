<!-- AUTO-GENERATED FILE — DO NOT EDIT DIRECTLY -->
<!-- Source of truth: src/variable_registry.py (VARIABLE_REGISTRY) -->
<!-- To update: edit the registry, then run `python generate_variable_docs.py` -->

# 95th Percentile Current Speed [m/s]

*Estimated extreme current speed, outlier-tolerant and comparable across sites for reconnaissance-level assessment*

## Description

95th percentile of the maximum current velocity magnitude across the water column calculated over the 1-year hindcast period, representing a robust, outlier-tolerant estimate of extreme current conditions at any depth under free-stream (undisturbed) conditions. This metric is intended for IEC 62600-201 [@iec_62600_201] Stage 1 reconnaissance-level analysis to provide a consistent basis for comparing extreme flow conditions across sites. It serves as a key input for initial structural loading assessment of turbine blades and towers, as well as for sizing electrical generation components and power electronics.

## Equation

$$
U_{95} = \text{percentile}(95, \left[\max(U_{1,t}, ..., U_{N_{\sigma},t}) \text{ for } t=1,...,T\right])
$$

**Where:**

- $U_{i,t} = \sqrt{u_{i,t}^2 + v_{i,t}^2}$, velocity magnitude at sigma layer $i$ at time $t$ $[\text{m/s}]$
- $\max_{\sigma}$, maximum value across all 10 sigma layers at each timestep
- $P_{95}$, 95th percentile operator over the full time series
- $N_{\sigma} = 10$, sigma layers
- $T$, 1 year of hindcast data (hourly for Alaska locations, half-hourly for others)

## Properties

| Property | Value |
| --- | --- |
| Internal Name | `vap_water_column_95th_percentile_sea_water_speed` |
| Units | m/s |

--8<-- "docs/tidal/high_resolution_hindcast/_cite-widget.md"
