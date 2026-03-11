<!-- AUTO-GENERATED FILE — DO NOT EDIT DIRECTLY -->
<!-- Source of truth: src/variable_registry.py (VARIABLE_REGISTRY) -->
<!-- To update: edit the registry, then run `python generate_variable_docs.py` -->

# 95th Percentile Current Speed [m/s]

*Estimated extreme current speed, outlier-tolerant and comparable across sites for reconnaissance-level assessment*

## Description

95th Percentile Current Speed provides a robust, outlier-tolerant estimate of extreme current conditions at each grid location, intended for consistent cross-site comparison during reconnaissance-level resource characterization. Unlike the absolute maximum, this statistic is resistant to isolated numerical artifacts and transient model effects, making it a more reliable and reproducible basis for comparing extreme flow conditions across sites.

This value is derived from a numerical hydrodynamic model and represents modeled conditions only. It should not be interpreted as a measured or ground-truth observation. Site-specific validation against in-situ measurements is recommended before use in detailed engineering design.

Engineering applications include preliminary structural loading assessment, blade and support structure design screening, and initial fatigue analysis.

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
