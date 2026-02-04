# WPTO High Resolution Tidal Hindcast

High-resolution 3D tidal current hindcast data for U.S. coastal locations, generated using the Finite Volume Community Ocean Model (FVCOM).

## Overview

This dataset contains high-resolution tidal hindcast data generated using FVCOM at five strategically selected U.S. coastal locations. The project represents a collaborative effort between [Pacific Northwest National Laboratory](https://www.pnnl.gov/marine-energy-resource-characterization) (PNNL) for data generation and the [National Laboratory of the Rockies](https://www.nlr.gov/water/resource-characterization) for data processing and visualization.

!!! info "Data Access"
Complete standardized datasets are available from the [AWS S3 Open Energy Data Initiative Marine Energy Data Lake](https://data.openei.org/s3_viewer?bucket=marine-energy-data). Summary data is visualized on the [Marine Energy Atlas](https://maps.nlr.gov/marine-energy-atlas).

!!! warning "Data Limitations"
This dataset is derived from numerical model simulations, not direct measurements. Results are based on a single hindcast year, which may not capture interannual variability in tidal energy resources. Model validation has been performed against observations at available measurement stations, but uncertainties exist, particularly in areas with complex bathymetry or limited observational data.

## Location Overview

| Location                        | Start Time [UTC]    | End Time [UTC]      | Sampling Frequency | Grid Points |
| ------------------------------- | ------------------- | ------------------- | ------------------ | ----------- |
| Aleutian Islands, Alaska        | 2010-06-03 00:00:00 | 2011-06-02 23:00:00 | Hourly             | 797,978     |
| Cook Inlet, Alaska              | 2005-01-01 00:00:00 | 2005-12-31 23:00:00 | Hourly             | 392,002     |
| Piscataqua River, New Hampshire | 2007-01-01 00:00:00 | 2007-12-31 23:30:00 | Half-Hourly        | 292,927     |
| Puget Sound, Washington         | 2015-01-01 00:00:00 | 2015-12-30 23:30:00 | Half-Hourly        | 1,734,765   |
| Western Passage, Maine          | 2017-01-01 00:00:00 | 2017-12-31 23:30:00 | Half-Hourly        | 231,208     |

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
| Boundary Forcing       | 12 tidal constituents from OSU TPXO Tide Models                             |
| Wind Forcing           | ERA5 or CFSv2                                                               |

## IEC Standards Compliance

| IEC Standards (IEC TS 62600-201, 2015)          | Compliance Status                            |
| ----------------------------------------------- | -------------------------------------------- |
| Stage 1 (feasibility) tidal resource assessment | Compliant with exceptions below              |
| Wave-current interaction                        | Not considered (waves small in study domain) |
| Atmospheric forcing                             | Not considered (effects negligible)          |
| Seawater density, salinity and temperature      | Not considered (density-induced flow small)  |

## Definitions

This section defines technical terms, mathematical symbols, and acronyms used throughout this document and in the dataset metadata.

### Technical Terms

**Hindcast**

A historical simulation of ocean conditions using a numerical model driven by observed atmospheric forcing and tidal boundary conditions. Unlike a forecast (which predicts future conditions) or direct measurements, a hindcast reconstructs past conditions by running the model over a historical time period. The results represent modeled estimates of what conditions were, not direct observations.

**Sigma Layer**

A terrain-following and free-surface-conforming vertical coordinate system used in ocean models where the water column is divided into layers that conform to both the fixed seafloor bathymetry (bottom boundary) and the time-varying sea surface (top boundary). Unlike fixed-depth coordinate systems, sigma (σ) expresses vertical position as a proportion of the instantaneous total water depth, ranging from σ = 0 at the free surface to σ = −1 at the seafloor. This approach maintains consistent vertical resolution regardless of local water depth, with layers stretching and compressing dynamically as the water column depth varies with tidal fluctuations.

Vertical Datum Convention: Sea surface elevation (ζ) is measured relative to NAVD88 and varies with time (tides, storm surge). Bathymetry depth (h) is the fixed depth of the seafloor below NAVD88. The instantaneous total water depth is D = h + ζ, and the physical depth below the surface for any sigma value is: depth = −D × σ.

Configuration in This Dataset: The FVCOM model uses 11 uniformly spaced sigma levels that define the horizontal boundaries of finite volume prisms. Between these levels are 10 sigma layers, representing the vertical midpoints of each prism. Model variables such as velocity components (u and v) are computed at each layer center (the centroid of the triangular prism at that sigma level). These are point values at the control volume center, not volume-integrated averages. The following tables detail the sigma coordinate values.

**FVCOM Sigma Levels (Boundaries)**

| Level Index | σ Value | Position    |
| ----------- | ------- | ----------- |
| 0           | 0.0     | Sea surface |
| 1           | −0.1    |             |
| 2           | −0.2    |             |
| 3           | −0.3    |             |
| 4           | −0.4    |             |
| 5           | −0.5    | Mid-depth   |
| 6           | −0.6    |             |
| 7           | −0.7    |             |
| 8           | −0.8    |             |
| 9           | −0.9    |             |
| 10          | −1.0    | Seafloor    |

**FVCOM Sigma Layers (Volume Centers)**

| Layer Index | σ Center | Description        |
| ----------- | -------- | ------------------ |
| 1           | −0.05    | Near-surface layer |
| 2           | −0.15    |                    |
| 3           | −0.25    |                    |
| 4           | −0.35    |                    |
| 5           | −0.45    |                    |
| 6           | −0.55    | Mid-depth layer    |
| 7           | −0.65    |                    |
| 8           | −0.75    |                    |
| 9           | −0.85    |                    |
| 10          | −0.95    | Near-bottom layer  |

How Layer Depths Vary with Tides: The physical depth of each sigma layer changes as the total water column depth varies with tides. The following table illustrates this using real data from Cook Inlet, Alaska (60.74°N, 151.43°W) where the bathymetry is h = 26.7 m below NAVD88. At low tide (ζ = -4.95 m, total depth D = 21.8 m) and high tide (ζ = 3.82 m, total depth D = 30.6 m), the sigma layer depths shift proportionally while maintaining their relative positions.

**Sigma Layer Depths at Low and High Tide (Cook Inlet Example)**

| Layer | σ Center | Low Tide (m) | High Tide (m) | Δ Depth (m) |
| ----- | -------- | ------------ | ------------- | ----------- |
| 1     | −0.05    | 1.1          | 1.5           | +0.4        |
| 2     | −0.15    | 3.3          | 4.6           | +1.3        |
| 3     | −0.25    | 5.4          | 7.6           | +2.2        |
| 4     | −0.35    | 7.6          | 10.7          | +3.1        |
| 5     | −0.45    | 9.8          | 13.7          | +3.9        |
| 6     | −0.55    | 12.0         | 16.8          | +4.8        |
| 7     | −0.65    | 14.2         | 19.9          | +5.7        |
| 8     | −0.75    | 16.3         | 22.9          | +6.6        |
| 9     | −0.85    | 18.5         | 26.0          | +7.5        |
| 10    | −0.95    | 20.7         | 29.0          | +8.3        |

Implications for Data Users: While sigma coordinates provide consistent data structure across varying bathymetry and tidal conditions, they require interpolation to extract data at fixed absolute depths. Queries such as 'average current speed at 8 m depth' require calculating which sigma layers correspond to that depth at each timestep.

**Depth-Averaged**

A value computed by averaging a quantity across all vertical layers (sigma layers) at a given horizontal location and time. Depth-averaging produces a single representative value for the entire water column, useful for characterizing overall flow conditions while accounting for vertical velocity profiles.

**Depth-Maximum**

The maximum value of a quantity across all vertical layers (sigma layers) at a given horizontal location and time. The depth-maximum captures the peak value occurring anywhere in the water column, which is relevant for structural loading calculations where components must withstand the highest forces at any depth.

**Free-Stream Velocity**

The undisturbed flow velocity that would exist in the absence of any energy extraction device. All velocity and power density values in this dataset represent free-stream (undisturbed) conditions and should not be used directly for turbine array yield estimation. Actual flow through turbine arrays will be modified by: (1) blockage effects that can reduce channel flow by 10-30% depending on the blockage ratio; (2) wake interactions where downstream turbines experience 20-40% velocity deficits in the near-wake region; and (3) device-induced turbulence that affects fatigue loading. Array yield calculations require site-specific wake modeling and cannot be derived directly from free-stream resource data.

**Unstructured Grid**

A computational mesh composed of irregularly-shaped elements (triangles in FVCOM) that can vary in size across the domain. This allows higher resolution in areas of interest (narrow channels, complex coastlines) and coarser resolution in open water, providing computational efficiency while maintaining accuracy where needed. For the duration of the model run, the horizontal coordinates (lat/lon) of grid remains fixed in space.

Grid Structure: The FVCOM model uses an unstructured triangular mesh where each element (cell) is defined by three nodes. Model variables are computed at element centers, while the mesh geometry is defined by node positions. The flexibility of triangular elements allows the mesh to conform to complex coastlines and bathymetric features without requiring uniform grid spacing.

Spatial Resolution: Grid resolution varies across the domain based on local requirements. In narrow channels and near coastlines where currents are strongest and bathymetry changes rapidly, triangles are smaller (higher resolution). In open water regions where conditions vary more gradually, triangles are larger (coarser resolution), reducing computational cost without sacrificing accuracy in areas of interest. The grid resolution variable in the dataset reports the average edge length of each triangular element.

**Model Limitations**

This dataset is derived from numerical model simulations and has inherent limitations that users should understand when applying the data.

Physics Not Included: The model does not include wave-current interaction (waves are small in the study domains), atmospheric forcing (wind and pressure effects on tidal currents are negligible), density-driven estuarine flow (temperature and salinity effects are small), or storm surge (only astronomical tidal forcing from 12 constituents via OSU TPXO Tide Models is applied).

Temporal Limitations: Results represent a single hindcast year and do not capture interannual variability. Tidal patterns are highly predictable, but extreme values may vary between years due to meteorological effects.

Spatial Limitations: The model uses wetting/drying algorithms for intertidal zones; results in these areas should be interpreted with care. Model accuracy may be reduced near open ocean boundaries. Features smaller than approximately 2-3 times the local grid spacing cannot be accurately resolved.

Known Data Gaps: Puget Sound is missing December 31, 2015. Specific files were excluded due to quality issues (see Quality Assurance section for details).

### Symbols

| Symbol | Definition                                                     |
| ------ | -------------------------------------------------------------- |
| U      | Current speed (velocity magnitude), m/s                        |
| Ū      | Mean (time-averaged) current speed, m/s                        |
| u      | Eastward velocity component, m/s (positive toward true east)   |
| v      | Northward velocity component, m/s (positive toward true north) |
| P      | Power density (kinetic energy flux per unit area), W/m²        |
| P̄      | Mean (time-averaged) power density, W/m²                       |
| ρ      | Seawater density, kg/m³ (nominal value: 1025 kg/m³)            |
| h      | Bathymetry depth below NAVD88, m (positive downward)           |
| ζ      | Sea surface elevation relative to NAVD88, m (positive upward)  |
| d      | Water depth (total water column height), m                     |
| R      | Grid resolution (average triangle edge length), m              |
| T      | Time period (hindcast duration = 1 year)                       |
| t      | Time index                                                     |
| i      | Sigma layer index (1 to Nσ)                                    |
| Nσ     | Number of sigma layers (= 10 in this dataset)                  |
| σ      | Sigma coordinate (terrain-following vertical coordinate)       |
| P₉₅    | 95th percentile operator                                       |

### Acronyms

| Acronym        | Definition                                                                       |
| -------------- | -------------------------------------------------------------------------------- |
| FVCOM          | Finite Volume Community Ocean Model                                              |
| NAVD88         | North American Vertical Datum of 1988                                            |
| MSL            | Mean Sea Level                                                                   |
| IEC            | International Electrotechnical Commission                                        |
| IEC 62600-201  | International standard for tidal energy resource assessment and characterization |
| CF Conventions | Climate and Forecast Conventions                                                 |
| PNNL           | Pacific Northwest National Laboratory                                            |
| WPTO           | Water Power Technologies Office                                                  |
| VAP            | Value-Added Product                                                              |

## Variable Documentation

### Variable Quick Reference

| Variable                                                        | Internal Name                                      | Units | Description                                                  |
| --------------------------------------------------------------- | -------------------------------------------------- | ----- | ------------------------------------------------------------ |
| [Mean Current Speed](#mean-current-speed)                       | `vap_water_column_mean_sea_water_speed`            | m/s   | Annual average of depth-averaged current speed               |
| [Mean Power Density](#mean-power-density)                       | `vap_water_column_mean_sea_water_power_density`    | W/m²  | Annual average of depth-averaged kinetic energy flux         |
| [95th Percentile Current Speed](#95th-percentile-current-speed) | `vap_water_column_95th_percentile_sea_water_speed` | m/s   | 95th percentile of yearly maximum current speed (all depths) |
| [Minimum Water Depth](#minimum-water-depth)                     | `vap_water_column_height_min`                      | m     | Minimum water depth (during 1 year model run)                |
| [Maximum Water Depth](#maximum-water-depth)                     | `vap_water_column_height_max`                      | m     | Maximum water depth (during 1 year model run)                |
| [Grid Resolution](#grid-resolution)                             | `vap_grid_resolution`                              | m     | Average edge length of triangular model grid cells           |

---

### Mean Current Speed {#mean-current-speed}

**Internal Name:** `vap_water_column_mean_sea_water_speed`
**Units:** m/s

> Annual average of depth-averaged current speed

#### Description

Mean Current Speed is the annual average of the depth-averaged current velocity magnitude, representing the characteristic flow speed at each grid location under free-stream (undisturbed) conditions. This metric is intended for IEC 62600-201 Stage 1 reconnaissance-level analysis to identify areas with tidal current resources. Calculated as the temporal mean of depth-averaged |U| where U = √(u² + v²).

Mean Current Speed is computed by first averaging velocity magnitudes across all 10 sigma layers at each timestep, then averaging over the full hindcast year. Current speed is the vector magnitude of eastward (u) and northward (v) velocity components: U = √(u² + v²). Tidal currents flow slower near the seafloor due to friction and faster in the upper water column. Depth-averaging provides a representative value for the entire water column.

Engineering applications include initial site screening, comparing relative site potential across regions, and Stage 1 IEC 62600-201 tidal energy resource characterization.

#### Equation

$$
\bar{\bar{U}} = \text{mean}\left(\left[\text{mean}(U_{1,t}, \ldots, U_{N_\sigma,t}) \text{ for } t=1,\ldots,T\right]\right)
$$

**Where:**

- $U_{i,t} = \sqrt{u^2 + v^2}$ — velocity magnitude at sigma layer $i$ at time $t$ (m/s)
- $u$ = eastward velocity component (m/s), positive toward true east
- $v$ = northward velocity component (m/s), positive toward true north
- $N_\sigma = 10$ sigma layers (terrain-following vertical layers)
- $T$ = 1 year of hindcast data

---

### Mean Power Density {#mean-power-density}

**Internal Name:** `vap_water_column_mean_sea_water_power_density`
**Units:** W/m²

> Annual average of depth-averaged kinetic energy flux

#### Description

Mean Power Density is the annual average of the kinetic energy flux per unit area, representing the theoretical power available for extraction from the undisturbed tidal flow. The cubic relationship with velocity (P = ½ρU³) makes this metric highly sensitive to current speed variations. Used for Stage 1 resource characterization and site ranking to indicate theoretical resource magnitude.

Power density is computed at each sigma layer using the cube of the current speed, then depth-averaged and temporally averaged over the full hindcast year. The cubic relationship with velocity means small increases in current speed yield large increases in available power—doubling the speed increases power density by a factor of eight.

Engineering applications include comparing relative energy availability between sites and initial economic feasibility screening.

#### Equation

$$
\bar{\bar{P}} = \text{mean}\left(\left[\text{mean}(P_{1,t}, \ldots, P_{N_\sigma,t}) \text{ for } t=1,\ldots,T\right]\right)
$$

**Where:**

- $P_{i,t} = \frac{1}{2}\rho U_{i,t}^3$ — power density at sigma layer $i$ at time $t$ (W/m²)
- $\rho = 1025$ kg/m³ (nominal seawater density)
- $U_{i,t} = \sqrt{u^2 + v^2}$ — velocity magnitude at sigma layer $i$ at time $t$ (m/s)
- $N_\sigma = 10$ sigma layers (terrain-following vertical layers)
- $T$ = 1 year of hindcast data

---

### 95th Percentile Current Speed {#95th-percentile-current-speed}

**Internal Name:** `vap_water_column_95th_percentile_sea_water_speed`
**Units:** m/s

> 95th percentile of yearly maximum current speed (all depths)

#### Description

95th Percentile Current Speed is the current speed exceeded only 5% of the time, computed from the depth-maximum (highest value across sigma layers) at each timestep. This metric characterizes extreme flow conditions relevant to structural loading and device survivability. Used for preliminary assessment of extreme current conditions in support of Stage 2 feasibility studies per IEC 62600-201. Calculated as P₉₅(max_σ(U)) where U = √(u² + v²).

At each timestep, the maximum speed across all 10 sigma layers is identified (the depth-maximum), then the 95th percentile of this time series is computed. The depth-maximum is used (rather than depth-average) because structural and mechanical systems must withstand peak loads that can occur at any depth in the water column. This metric characterizes expected high-flow conditions for survivability design, while excluding rare extreme events.

Engineering applications include structural loading calculations, blade and support structure design, and fatigue analysis. This metric helps size components to withstand expected high-flow conditions.

#### Equation

$$
U_{95} = P_{95}\left(\left[\max(U_{1,t}, \ldots, U_{N_\sigma,t}) \text{ for } t=1,\ldots,T\right]\right)
$$

**Where:**

- $U_{i,t} = \sqrt{u^2 + v^2}$ — velocity magnitude at sigma layer $i$ at time $t$ (m/s)
- $\max_\sigma$ = maximum value across all sigma layers at each timestep
- $P_{95}$ = 95th percentile operator over the time series
- $N_\sigma = 10$ sigma layers (terrain-following vertical layers)
- $T$ = 1 year of hindcast data

---

### Minimum Water Depth {#minimum-water-depth}

**Internal Name:** `vap_water_column_height_min`
**Units:** m

> Minimum water depth (during 1 year model run)

#### Description

Minimum Water Depth is the lowest water depth (surface to seafloor) observed at each grid location over the hindcast year, typically occurring during extreme low tide conditions. This metric defines the minimum vertical clearance available for device deployment and is critical for assessing depth constraints. Used in Stage 2 feasibility studies for turbine placement and collision avoidance.

Minimum water depth typically occurs during spring tides when tidal range is maximized. Total water depth is calculated as the sum of bathymetry depth (h) and sea surface elevation (ζ). The difference between maximum and minimum water depth approximates the tidal range at each location.

Engineering applications include assessing turbine clearance requirements and identifying areas where shallow water may limit device deployment.

#### Equation

$$
d_{\min} = \min_t(h + \zeta_t)
$$

**Where:**

- $h$ = bathymetry depth below NAVD88 (m)
- $\zeta_t$ = sea surface elevation above NAVD88 at time $t$ (m)
- $T$ = 1 year of hindcast data

---

### Maximum Water Depth {#maximum-water-depth}

**Internal Name:** `vap_water_column_height_max`
**Units:** m

> Maximum water depth (during 1 year model run)

#### Description

Maximum Water Depth is the greatest water depth (surface to seafloor) observed at each grid location over the hindcast year, typically occurring during extreme high tide conditions. This metric represents the upper bound of water depth variability at each location. Used in Stage 2 feasibility studies for mooring system design, cable routing, and understanding the full operating depth envelope.

Maximum water depth typically occurs during spring tides when tidal range is maximized. Water depth is calculated as the sum of bathymetry depth (h) and sea surface elevation (ζ). The difference between maximum and minimum water depth approximates the tidal range at each location.

Engineering applications include mooring system design considerations and understanding the full range of water depths at a site.

#### Equation

$$
d_{\max} = \max_t(h + \zeta_t)
$$

**Where:**

- $h$ = bathymetry depth below NAVD88 (m)
- $\zeta_t$ = sea surface elevation above NAVD88 at time $t$ (m)
- $T$ = 1 year of hindcast data

---

### Grid Resolution {#grid-resolution}

**Internal Name:** `vap_grid_resolution`
**Units:** m

> Average edge length of triangular model grid cells

#### Description

Grid Resolution is the average edge length of the unstructured triangular model grid cells, indicating the spatial scale at which tidal currents are resolved by the FVCOM hydrodynamic model. Essential model metadata for assessing spatial uncertainty and determining appropriate applications. IEC 62600-201 requires <500 m for Stage 1 reconnaissance and <50 m for Stage 2 feasibility assessments.

Grid Resolution is calculated as the average of the three edge lengths for each triangular grid cell, based on the original unstructured mesh defined by the model developers at Pacific Northwest National Laboratory. The unstructured triangular mesh allows variable resolution, with finer grids in areas of interest (channels, straits) and coarser grids in open water.

Per IEC 62600-201 tidal energy resource assessment standards:

- Stage 1 feasibility (reconnaissance-level) assessments require grid resolution < 500 m
- Stage 2 (layout design) assessments require grid resolution < 50 m

Engineering applications include assessing model fidelity and determining appropriate applications for the data.

#### Equation

$$
R = \frac{1}{3}(d_1 + d_2 + d_3)
$$

**Where:**

- $d_1, d_2, d_3$ = geodesic distances between triangle vertices (m)

---

## Data Formats and Levels

| Data Level                     | Description                                 | Format  | Public Access |
| ------------------------------ | ------------------------------------------- | ------- | ------------- |
| `00_raw`                       | Original FVCOM model outputs                | NetCDF  | No            |
| `a1_std`                       | Standardized with consistent naming         | NetCDF  | No            |
| `b1_vap`                       | Value-added products with derived variables | NetCDF  | **Yes**       |
| `b4_vap_summary_parquet`       | Summary statistics for analytics            | Parquet | **Yes**       |
| `b5_vap_atlas_summary_parquet` | Marine Energy Atlas visualization data      | Parquet | **Yes**       |

## Data Access

### Python with rex

The easiest way to access the data is using the [rex](https://github.com/nlr/rex) library:

```python
from rex import ResourceX

# Access tidal data via HSDS
tidal_file = '/nlr/US_tidal/Cook_Inlet/Cook_Inlet_2005.h5'

with ResourceX(tidal_file, hsds=True) as f:
    meta = f.meta
    time_index = f.time_index
    speed = f['sea_water_speed']
```

### HSDS Configuration

To use HSDS, first install and configure h5pyd:

```bash
pip install h5pyd
hsconfigure
```

Enter the following at the prompt:

```
hs_endpoint = https://developer.nlr.gov/api/hsds
hs_username =
hs_password =
hs_api_key = YOUR_API_KEY
```

!!! warning "API Key Required"
Get your own API key at [https://developer.nlr.gov/signup/](https://developer.nlr.gov/signup/). The example key is rate-limited.

### Direct h5pyd Access

```python
import h5pyd
import pandas as pd

with h5pyd.File('/nlr/US_tidal/Cook_Inlet/Cook_Inlet_2005.h5', mode='r') as f:
    meta = pd.DataFrame(f['meta'][...])
    speed = f['sea_water_speed']
    scale_factor = speed.attrs['scale_factor']
    mean_speed = speed[...].mean(axis=0) / scale_factor

meta['Mean Speed'] = mean_speed
```

### Extract Time Series at a Location

```python
from rex import ResourceX

tidal_file = '/nlr/US_tidal/Cook_Inlet/Cook_Inlet_2005.h5'
lat_lon = (60.5, -151.5)  # Cook Inlet location

with ResourceX(tidal_file, hsds=True) as f:
    speed_ts = f.get_lat_lon_df('sea_water_speed', lat_lon)
```

## Quality Assurance

### Verification Process

| Category             | Verification Process                                | Status    |
| -------------------- | --------------------------------------------------- | --------- |
| Model Specification  | FVCOM version, CF conventions, required variables   | Compliant |
| Temporal Integrity   | Time steps, chronological order, expected frequency | Compliant |
| Spatial Consistency  | Coordinate systems, grid consistency                | Compliant |
| Metadata Consistency | Global attributes across files                      | Compliant |
| Dataset Structure    | Dimensions, variables, coordinates                  | Compliant |
| Completeness         | Temporal coverage, required variables               | Compliant |

### Known Data Gaps

| Location         | Issue                                     |
| ---------------- | ----------------------------------------- |
| Puget Sound      | Missing 2015-12-31                        |
| Aleutian Islands | File MD_AIS_west_hrBathy_0370.nc excluded |
| Cook Inlet       | File cki_0366.nc excluded                 |
| Piscataqua River | File PIR_0368.nc excluded                 |

## Coordinate Systems

| Location         | Original System      | Transformation |
| ---------------- | -------------------- | -------------- |
| Aleutian Islands | Geographic (lat/lon) | None           |
| Cook Inlet       | Geographic (lat/lon) | None           |
| Western Passage  | UTM Zone 19 (NAD83)  | → WGS84        |
| Piscataqua River | UTM Zone 19 (NAD83)  | → WGS84        |
| Puget Sound      | UTM Zone 10 (NAD83)  | → WGS84        |

All data is transformed to WGS84 (EPSG:4326) for consistency.

## References

### Standards and Model Documentation

**(International Electrotechnical Commission 2015)** International Electrotechnical Commission. 2015. 'Marine energy – Wave, tidal and other water current converters – Part 201: Tidal energy resource assessment and characterization'. 1.0. Geneva, Switzerland.

**(Chen, Beardsley, and Cowles 2006)** Chen, Changsheng, Robert C. Beardsley, and Geoffrey Cowles. 2006. 'An Unstructured Grid, Finite-Volume Coastal Ocean Model (FVCOM) System'. Oceanography 19 (1): 78–89. https://doi.org/10.5670/oceanog.2006.92.

### Location-Specific Validation Studies

**Alaska, Aleutian Islands**

- Spicer, Preston, Zhaoqing Yang, Taiping Wang, and Mithun Deb. 2025. 'Spatially Varying Seasonal Modulation to Tidal Stream Energy Potential Due to Mixed Tidal Regimes in the Aleutian Islands, AK'. Renewable Energy 253:123564. https://doi.org/10.1016/j.renene.2025.123564.

**Alaska, Cook Inlet**

- Deb, Mithun, Zhaoqing Yang, and Taiping Wang. 2025. 'Characterizing In-stream Turbulent Flow for Tidal Energy Converter Siting in Cook Inlet, Alaska'. Renewable Energy 252 (May):123345. https://doi.org/10.1016/j.renene.2025.123345.

**Maine, Western Passage**

- Deb, Mithun, Zhaoqing Yang, Taiping Wang, and Levi Kilcher. 2023. 'Turbulence Modeling to Aid Tidal Energy Resource Characterization in the Western Passage, Maine, USA'. Renewable Energy 219 (April). https://doi.org/10.1016/j.renene.2023.04.100.
- Yang, Zhaoqing, Taiping Wang, Ziyu Xiao, Levi Kilcher, Kevin Haas, Huijie Xue, and Xi Feng. 2020. 'Modeling Assessment of Tidal Energy Extraction in the Western Passage'. Journal of Marine Science and Engineering 8 (6). https://doi.org/10.3390/jmse8060411.

**New Hampshire, Piscataqua River**

- Spicer, Preston, Zhaoqing Yang, Taiping Wang, and Mithun Deb. 2023. 'Tidal Energy Extraction Modifies Tidal Asymmetry and Transport in a Shallow, Well-mixed Estuary'. Frontiers in Marine Science 10 (September). https://doi.org/10.3389/fmars.2023.1268348.

**Washington, Puget Sound**

- Deb, Mithun, Zhaoqing Yang, Kevin Haas, and Taiping Wang. 2024. 'Hydrokinetic Tidal Energy Resource Assessment Following International Electrotechnical Commission Guidelines'. Renewable Energy 229 (June):120767. https://doi.org/10.1016/j.renene.2024.120767.
- Spicer, Preston, Parker Maccready, and Zhaoqing Yang. 2024. 'Localized Tidal Energy Extraction in Puget Sound Can Adjust Estuary Reflection and Friction, Modifying Barotropic Tides System-Wide'. Journal of Geophysical Research: Oceans 129 (May). https://doi.org/10.1029/2023JC020401.
- Yang, Zhaoqing, Taiping Wang, Ruth Branch, Ziyu Xiao, and Mithun Deb. 2021. 'Tidal Stream Energy Resource Characterization in the Salish Sea'. Renewable Energy 172 (March). https://doi.org/10.1016/j.renene.2021.03.028.

## Citation

Yang, Zhaoqing, Mithun Deb, Taiping Wang, Preston Spicer, Andrew Simms, Ethan Young, and Mike Lawson. 2025. "High Resolution Tidal Hindcast."

## Acknowledgement

This work was funded by the U.S. Department of Energy, Office of Energy Efficiency & Renewable Energy, Water Power Technologies Office. The authors gratefully acknowledge project support from Heather Spence and Jim McNally (U.S. Department of Energy Water Power Technologies Office) and Mary Serafin (National Laboratory of the Rockies). Technical guidance was provided by Levi Kilcher, Caroline Draxl, and Katie Peterson (National Laboratory of the Rockies).

## Disclaimer

This data arose from work performed under funding provided by the United States Government. Access to or use of this data denotes consent with the fact that this data is provided "AS IS," "WHERE IS" and specifically free from any express or implied warranty of any kind, including but not limited to any implied warranties such as merchantability and/or fitness for any particular purpose.

The user is granted the right, without any fee or cost, to use or copy the Data, provided that this entire notice appears in all copies. Users engaging in scientific or technical publication utilizing this data agree to credit DOE/PNNL/NLR/BATTELLE/ALLIANCE consistent with professional practice.
