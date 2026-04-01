# Methodology

## Pipeline Overview

The data processing pipeline transforms raw FVCOM model outputs into standardized, publicly accessible datasets:

```
Raw FVCOM Output → Verification → Standardization → VAP Derivation → Summary Statistics → Atlas Data
```

## Raw Data (Level 00)

Original FVCOM NetCDF outputs containing:

- 3D velocity fields (u, v components) at 10 sigma layers
- Sea surface elevation (zeta)
- Bathymetry
- Grid connectivity (node/element relationships)

## Standardization (Level a1)

Raw outputs are standardized with:

- Consistent variable naming following CF Conventions
- Coordinate transformation to WGS84 (EPSG:4326) where necessary (see [Coordinate Systems](coordinate-systems.md))
- Temporal validation and formatting to ISO 8601
- Global attribute metadata following ACDD conventions

## VAP Derivation (Level b1)

Value-Added Products (VAPs) are computed from the standardized data. Each variable in the [Variables](variables/index.md) section documents its specific derivation equation. General methods include:

### Depth-Averaging

For mean variables (current speed, power density), values are averaged across all 10 sigma layers at each timestep, then averaged over the full hindcast year:

$$
\overline{X} = \text{mean}_t\left(\text{mean}_\sigma(X_{i,t})\right)
$$

### Depth-Maximum

For 95th percentile variables, the maximum value across sigma layers is taken at each timestep, then the 95th percentile is computed over the full time series:

$$
X_{95} = P_{95}\left(\max_\sigma(X_{i,t})\right)
$$

### Current Speed

Velocity magnitude is computed from eastward ($u$) and northward ($v$) components:

$$
U = \sqrt{u^2 + v^2}
$$

### Power Density

Kinetic energy flux per unit area [@hass_2011_assessment]:

$$
P = \frac{1}{2} \rho U^3
$$

where $\rho = 1025$ kg/m$^3$ (nominal seawater density).

## Summary Statistics (Level b4)

Summary-level Parquet files contain one row per grid point with all VAP variables. These enable efficient analytical queries without loading full time-series data.

## Atlas Data (Level b5)

A subset of summary statistics formatted for the [Marine Energy Atlas](../../getting-started/marine-energy-atlas.md) visualization platform, including spatial metadata for map rendering.

## Quality Control

Quality control is applied at each processing stage. See [Quality Assurance](quality-assurance.md) for verification details.


--8<-- "docs/tidal/high_resolution_hindcast/_cite-widget.md"
