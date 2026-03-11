<!-- AUTO-GENERATED FILE — DO NOT EDIT DIRECTLY -->
<!-- Source of truth: src/variable_registry.py (VARIABLE_REGISTRY) -->
<!-- To update: edit the registry, then run `python generate_variable_docs.py` -->

# 95th Percentile Power Density [W/m²]

*Estimated extreme power density, outlier-tolerant and robust to cubic-velocity sensitivity*

## Description

95th percentile of the maximum power density (kinetic energy flux) across the water column calculated over the 1-year hindcast period, representing a robust, outlier-tolerant estimate of extreme energy flux at any depth under free-stream (undisturbed) conditions. Due to the cubic relationship between velocity and power density, extreme values are particularly sensitive to model artifacts, making the 95th percentile a more reliable metric than the absolute maximum for comparing extreme energy flux across sites. It is intended for IEC 62600-201 [@iec_62600_201] Stage 1 reconnaissance-level analysis and is intended to be input for extreme load analysis on tidal turbine components.

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
