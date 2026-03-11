<!-- AUTO-GENERATED FILE — DO NOT EDIT DIRECTLY -->
<!-- Source of truth: src/variable_registry.py (VARIABLE_REGISTRY) -->
<!-- To update: edit the registry, then run `python generate_variable_docs.py` -->

# Tidal Range [m]

*Difference between maximum and minimum sea surface elevation over the hindcast year*

## Description

Tidal Range is the difference between the maximum and minimum sea surface elevation observed at each grid location over the hindcast year. This metric characterizes the tidal regime and vertical water level variability. IEC 62600-201 requires tidal range description as part of Stage 1 site characterization. Tidal range is classified as microtidal (<2m), mesotidal (2-4m), or macrotidal (>4m). Larger tidal ranges indicate stronger tidal forcing, which generally correlates with stronger tidal currents but also creates greater challenges for device deployment due to water level variability. Engineering applications include mooring system design (must accommodate full range of water levels), cable routing, and understanding the relationship between water level and current speed at a site.

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
