# FVCOM Configuration

## Model Configuration

| Configuration          | Specification                                                               |
| ---------------------- | --------------------------------------------------------------------------- |
| Model                  | [Finite Volume Community Ocean Model (FVCOM) 4.3.1](https://www.fvcom.org)  |
| Dimensions             | time, element, depth                                                        |
| Horizontal Resolution  | Unstructured triangular grid (50-500m for IEC 62600-201 Stage 1 compliance) |
| Vertical Resolution    | 10 uniform sigma layers from surface to seafloor                            |
| Horizontal Coordinates | Latitude, Longitude (EPSG:4326)                                             |
| Vertical Datum         | Mean sea level (MSL), NAVD88                                                |
| Temporal Resolution    | 1 year at half-hourly or hourly intervals                                   |
| Wetting & Drying       | Activated                                                                   |
| Boundary Forcing       | 12 tidal constituents from OSU TPXO Tide Models                            |
| Wind Forcing           | ERA5 or CFSv2                                                               |

## FVCOM Overview

The [Finite Volume Community Ocean Model](http://fvcom.smast.umassd.edu/fvcom) (FVCOM) is an unstructured-grid, finite-volume, three-dimensional ocean circulation model. Key features relevant to this dataset:

- **Unstructured triangular mesh** allows variable spatial resolution, concentrating grid points in areas of interest (narrow channels, complex coastlines) while using coarser resolution in open water
- **Sigma coordinate system** provides terrain-following vertical layers that adapt to local water depth
- **Finite volume method** conserves mass, momentum, and energy across the domain
- **Wetting/drying** algorithm handles intertidal zones where the seafloor is alternately submerged and exposed

## Forcing and Boundary Conditions

### Tidal Forcing

The model is driven by 12 tidal constituents from the [OSU TPXO Tide Models](https://www.tpxo.net/) applied at open ocean boundaries. This provides the astronomical tidal forcing that drives the primary flow patterns.

### Wind Forcing

Surface wind stress is applied using ERA5 or CFSv2 reanalysis data, depending on the location and time period. Wind effects on tidal currents in these study domains are generally small.

### Physics Not Included

- **Wave-current interaction** — Waves are small in all study domains
- **Atmospheric pressure forcing** — Effects on tidal currents are negligible
- **Density-driven flow** — Temperature and salinity effects are small in these tidally dominated environments
- **Storm surge** — Only astronomical tidal forcing is applied

See [Limitations](limitations.md) for a complete discussion of model limitations.


--8<-- "docs/tidal/high_resolution_hindcast/_cite-widget.md"
