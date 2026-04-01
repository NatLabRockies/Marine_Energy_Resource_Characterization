<!-- AUTO-GENERATED FILE — DO NOT EDIT DIRECTLY -->
<!-- Source of truth: src/variable_registry.py (VARIABLE_REGISTRY) -->
<!-- To update: edit the registry, then run `python generate_variable_docs.py` -->

# Tidal Range [m]

*Difference between maximum and minimum sea surface elevation over the hindcast year*

## Description

Difference between the maximum and minimum sea surface elevation calculated at each grid location over the 1-year hindcast period. This metric quantifies the full vertical extent of water level variability at a site.

## Equation

$$
R = \zeta_{\max} - \zeta_{\min} = \max(\zeta_t) - \min(\zeta_t)
$$

**Where:**

- $\zeta_t$, sea surface elevation at time $t$ $[\text{m}]$
- $\zeta_{\max}$, maximum sea surface elevation over the year $[\text{m}]$
- $\zeta_{\min}$, minimum sea surface elevation over the year $[\text{m}]$
- $T$, 1 year of hindcast data

## Properties

| Property | Value |
| --- | --- |
| Internal Name | `vap_tidal_range` |
| Units | m |

--8<-- "docs/tidal/high_resolution_hindcast/_cite-widget.md"
