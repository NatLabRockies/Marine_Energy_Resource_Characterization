<!-- AUTO-GENERATED FILE — DO NOT EDIT DIRECTLY -->
<!-- Source of truth: src/variable_registry.py (VARIABLE_REGISTRY) -->
<!-- To update: edit the registry, then run `python generate_variable_docs.py` -->

# 95th Percentile Power Density [W/m²]

*Estimated extreme power density, outlier-tolerant and comparable across sites for reconnaissance-level assessment*

## Description

95th Percentile Power Density provides a robust, outlier-tolerant estimate of extreme power density conditions at each grid location, intended for consistent cross-site comparison during reconnaissance-level resource characterization. Unlike the absolute maximum, this statistic is resistant to isolated numerical artifacts and transient model effects. Due to the cubic relationship between velocity and power density, extreme values are particularly sensitive to model artifacts, making the 95th percentile a more reliable and reproducible basis for comparing extreme energy flux across sites.

This value is derived from a numerical hydrodynamic model and represents modeled conditions only. It should not be interpreted as a measured or ground-truth observation. Site-specific validation against in-situ measurements is recommended before use in detailed engineering design.

Engineering applications include preliminary extreme load assessment, power electronics sizing, and initial design margin estimation.

## Equation

$$
P_{95} = \text{percentile}(95, \left[\max(P_{1,t}, ..., P_{N_{\sigma},t}) \text{ for } t=1,...,T\right])
$$

**Where:**

- $P_{i,t} = \frac{1}{2} \rho U_{i,t}^3$, power density with $\rho = 1025$ $[\text{kg/m}^3]$
- $U_{i,t} = \sqrt{u_{i,t}^2 + v_{i,t}^2}$, velocity magnitude at sigma level $i$ at time $t$ $[\text{m/s}]$
- $N_{\sigma} = 10$, sigma layers
- $T$, 1 year of hindcast data

## Properties

| Property | Value |
| --- | --- |
| Internal Name | `vap_water_column_95th_percentile_sea_water_power_density` |
| Units | W/m² |

--8<-- "docs/tidal/high_resolution_hindcast/_cite-widget.md"
