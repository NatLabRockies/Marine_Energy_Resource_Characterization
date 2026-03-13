<!-- AUTO-GENERATED FILE — DO NOT EDIT DIRECTLY -->
<!-- Source of truth: src/variable_registry.py (VARIABLE_REGISTRY) -->
<!-- To update: edit the registry, then run `python generate_variable_docs.py` -->

# Mean Power Density [W/m²]

*Annual average of depth-averaged kinetic energy flux*

## Description

Annual average of the kinetic energy flux per unit area (depth-averaged) calculated over the 1-year hindcast period, representing a mean estimate of energy flux for the entire water column at each grid location under free-stream (undisturbed) conditions. This metric is intended for IEC 62600-201 [@iec_62600_201] Stage 1 reconnaissance-level analysis to identify areas with potentially viable tidal current resources. This metric defines the theoretic average energy potential based on model output and does not account for changes in salinity, temperature, or turbulence that may affect real-world energy extraction.

## Equation

$$
\overline{\overline{P}} = P_{\text{average}} = \text{mean}\left(\left[\text{mean}(P_{1,t}, ..., P_{N_{\sigma},t}) \text{ for } t=1,...,T\right]\right)
$$

**Where:**

- $P_{i,t} = \frac{1}{2} \rho U_{i,t}^3$, power density at sigma layer $i$ at time $t$ $[\text{W/m}^2]$ [@hass_2011_assessment]
- $\rho = 1025$, nominal seawater density (actual varies with temperature and salinity) $[\text{kg/m}^3]$
- $U_{i,t} = \sqrt{u_{i,t}^2 + v_{i,t}^2}$, velocity magnitude $[\text{m/s}]$
- $N_{\sigma} = 10$, sigma layers
- $T$, 1 year of hindcast data (hourly for Alaska locations, half-hourly for others)

## Properties

| Property | Value |
| --- | --- |
| Internal Name | `vap_water_column_mean_sea_water_power_density` |
| Units | W/m² |

--8<-- "docs/tidal/high_resolution_hindcast/_cite-widget.md"
