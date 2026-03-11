<!-- AUTO-GENERATED FILE — DO NOT EDIT DIRECTLY -->
<!-- Source of truth: src/variable_registry.py (VARIABLE_REGISTRY) -->
<!-- To update: edit the registry, then run `python generate_variable_docs.py` -->

# Mean Power Density [W/m²]

*Annual average of depth-averaged kinetic energy flux*

## Description

Mean Power Density is the annual average of the kinetic energy flux per unit area, representing the theoretical power available for extraction from the undisturbed tidal flow. The cubic relationship with velocity makes this metric highly sensitive to current speed variations. Used for Stage 1 resource characterization and site ranking to indicate theoretical resource magnitude. Engineering applications include comparing relative energy availability between sites and initial economic feasibility screening.

## Equation

$$
\overline{\overline{P}} = P_{\text{average}} = \text{mean}\left(\left[\text{mean}(P_{1,t}, ..., P_{N_{\sigma},t}) \text{ for } t=1,...,T\right]\right)
$$

**Where:**

- $P_{i,t} = \frac{1}{2} \rho U_{i,t}^3$, power density at sigma layer $i$ at time $t$ $[\text{W/m}^2]$
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
