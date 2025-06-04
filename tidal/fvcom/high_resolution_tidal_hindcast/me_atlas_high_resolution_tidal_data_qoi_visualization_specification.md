# ME Atlas High Resolution Tidal Data QOI Visualization Specification

The following sections provide the specification for visualizing selected high resolution tidal hindcast variables on the [NREL Marine Energy Atlas](https://maps.nrel.gov/marine-energy-atlas/data-viewer/data-library/layers?vL=WavePowerMerged)

## Available Data File Details

Base directory for all data files:

- <base_dir>: `/projects/hindcastra/Tidal/datasets/high_resolution_tidal_hindcast`

| Location Name                   | System           | File Path                                                                                                                                      |
| ------------------------------- | ---------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| Puget Sound, Washington         | NREL Kestrel HPC | `<base_dir>/WA_puget_sound/b6_vap_atlas_summary_parquet/WA_puget_sound.tidal_hindcast_fvcom-year_average.b5.20150101.000000.parquet`           |
| Piscataqua River, New Hampshire | NREL Kestrel HPC | `<base_dir>/NH_piscataqua_river/b6_vap_atlas_summary_parquet/NH_piscataqua_river.tidal_hindcast_fvcom-year_average.b5.20070101.000000.parquet` |
| Western Passage, Maine          | NREL Kestrel HPC | `<base_dir>/ME_western_passage/b6_vap_atlas_summary_parquet/ME_western_passage.tidal_hindcast_fvcom-year_average.b5.20170101.000000.parquet`   |
| Cook Inlet, Alaska              | NREL Kestrel HPC | `<base_dir>/AK_cook_inlet/b6_vap_atlas_summary_parquet/AK_cook_inlet.tidal_hindcast_fvcom-year_average.b5.20050101.000000.parquet`             |
| Aleutian Islands, Alaska        | NREL Kestrel HPC | `<base_dir>/AK_aleutian_islands/b6_vap_atlas_summary_parquet/AK_aleutian_islands.tidal_hindcast_fvcom-year_average.b5.20100603.000000.parquet` |

## Location Details

| Location Name                   | Face Count | Averaging Dates [UTC]                      | Averaging Temporal Resolution |
| ------------------------------- | ---------- | ------------------------------------------ | ----------------------------- |
| Puget Sound, Washington         | 1,734,765  | 2015-01-01 00:00:00 to 2015-12-30 23:30:00 | half-hourly                   |
| Piscataqua River, New Hampshire | 292,927    | 2007-01-01 00:00:00 to 2007-12-31 23:30:00 | half-hourly                   |
| Western Passage, Maine          | 231,208    | 2017-01-01 00:00:00 to 2017-12-31 23:30:00 | half-hourly                   |
| Cook Inlet, Alaska              | 392,002    | 2005-01-01 00:00:00 to 2005-12-31 23:00:00 | hourly                        |
| Aleutian Islands, Alaska        | 797,978    | 2010-06-03 00:00:00 to 2011-06-02 23:00:00 | hourly                        |

## Variable Overview

| Variable                                | Units            | Data Column                                              |
| --------------------------------------- | ---------------- | -------------------------------------------------------- |
| Mean Sea Water Speed                    | m/s              | vap_water_column_mean_sea_water_speed                    |
| 95th Percentile Sea Water Speed         | m/s              | vap_water_column_95th_percentile_sea_water_speed         |
| Mean Sea Water Power Density            | W/m²             | vap_water_column_mean_sea_water_power_density            |
| 95th Percentile Sea Water Power Density | W/m²             | vap_water_column_95th_percentile_sea_water_power_density |
| Mean Depth                              | m (below NAVD88) | vap_sea_floor_depth                                      |

## Variable Usage

| Variable                                | Meaning                                                                                     | Intended Usage                                            |
| --------------------------------------- | ------------------------------------------------------------------------------------------- | --------------------------------------------------------- |
| Mean Sea Water Speed                    | Yearly average of depth averaged current speed                                              | Site screening and turbine selection for power generation |
| 95th Percentile Sea Water Speed         | 95th percentile of yearly depth maximum current speed                                       | Generator sizing and power electronics design             |
| Mean Sea Water Power Density            | Yearly average of depth averaged power density (kinetic energy flux)                        | Resource quantification and economic feasibility analysis |
| 95th Percentile Sea Water Power Density | 95th percentile of the yearly maximum of depth averaged power density (kinetic energy flux) | Structural design loads and extreme loading conditions    |
| Mean Depth                              | Yearly average distance from water surface to the sea floor                                 | Installation planning and foundation design               |

## Variable Equations

### Mean Sea Water Speed

Equation:

$\overline{\overline{U}} = U_{\text{average}} = \text{mean}\left(\left[\text{mean}(U_{1,t}, ..., U_{N_{\sigma},t}) \text{ for } t=1,...,T\right]\right)$

Where:$U_{i,t} = \sqrt{u_{i,t}^2 + v_{i,t}^2}$ are velocity magnitudes at uniformly distributed sigma level $i$ at volume centers at time $t$ (m/s), $N_{\sigma} = 10$ levels, $T = 1$ year

### 95th Percentile Sea Water Speed

Equation:

$U_{95} = \text{percentile}_{95}\left(\left[\max(U_{1,t}, ..., U_{N_{\sigma},t}) \text{ for } t=1,...,T\right]\right)$

Where: $U_{i,t} = \sqrt{u_{i,t}^2 + v_{i,t}^2}$ are velocity magnitudes at uniformly distributed sigma level $i$ at volume centers at time $t$ (m/s), $N_{\sigma} = 10$ levels, $T = 1$ year

### Mean Sea Water Power Density

Equation:

$\overline{\overline{P}} = P_{\text{average}} = \text{mean}\left(\left[\text{mean}(P_{1,t}, ..., P_{N_{\sigma},t}) \text{ for } t=1,...,T\right]\right)$

Where $P_{i,t} = \frac{1}{2} \rho U_{i,t}^3$ with $\rho = 1025$ kg/m³, $U_{i,t}$ are velocity magnitudes at uniformly distributed sigma level $i$ at volume centers at time $t$, $N_{\sigma} = 10$ levels, $T = 1$ year

### 95th Percentile Sea Water Power Density

Equation:

$P_{95} = \text{percentile}_{95}\left(\left[\max(P_{1,t}, ..., P_{N_{\sigma},t}) \text{ for } t=1,...,T\right]\right)$

Where $P_{i,t} = \frac{1}{2} \rho U_{i,t}^3$ with $\rho = 1025$ kg/m³, $U_{i,t}$ are velocity magnitudes at uniformly distributed sigma level $i$ at volume centers at time $t$, $N_{\sigma} = 10$ levels, $T = 1$ year

### Mean Depth

Equation:

$\overline{d} = d_{\text{average}} = \text{mean}\left(\left[(h + \zeta_t) \text{ for } t=1,...,T\right]\right)$

Where $h$ is bathymetry below NAVD88 (m), $\zeta_t$ is sea surface elevation above NAVD88 at time $t$ (m), $T = 1$ year

## Coordinate Details

The high resolution tidal hindcast data is based on an unstructured three dimensional grid of triangular faces with variable resolution.
To visualize in two dimensions (lat/lon) the data for all depths are combined (averaging, or 95th percentile of maximums) into a single layer.
This single layer has coordinates defined at the center and corners of each triangular element.
Within the parquet files the coordinates are stored in the following columns:

Notes:

- All coordinates are in WGS84 (EPSG:4326) format.
- All centerpoints have been validated to be within the bounding box of the triangular element.
- All triangular elements coordinates are visualized below and can be assumed to be valid
- Triangular elements are not guaranteed to be equilateral or isosceles, and may have varying angles and lengths.
- Triangular elements vertice order has not been validated to be consistent across all regions.
- The Aleutian Islands, Alaska dataset has elements that cross the from -180 to 180 longitude, which may cause visual artifacts in some mapping software.

| Column Name            | Description                            |
| ---------------------- | -------------------------------------- |
| `lat_center`           | Element Center Latitude                |
| `lon_center`           | Element Center Longitude               |
| `element_corner_1_lat` | Element Triangular Vertice 1 Latitude  |
| `element_corner_1_lon` | Element Triangular Vertice 1 Longitude |
| `element_corner_2_lat` | Element Triangular Vertice 2 Latitude  |
| `element_corner_2_lon` | Element Triangular Vertice 2 Longitude |
| `element_corner_3_lat` | Element Triangular Vertice 3 Latitude  |
| `element_corner_3_lon` | Element Triangular Vertice 3 Longitude |

## Color Details

| Variable                                | Column Name                                                | Range     | Units            | Discrete Levels | Colormap        |
| --------------------------------------- | ---------------------------------------------------------- | --------- | ---------------- | --------------- | --------------- |
| Mean Sea Water Speed                    | `vap_water_column_mean_sea_water_speed`                    | 0.0 - 1.5 | m/s              | 10              | cmocean.thermal |
| 95th Percentile Sea Water Speed         | `vap_water_column_95th_percentile_sea_water_speed`         | 0.0 - 4.0 | m/s              | 8               | cmocean.matter  |
| Mean Sea Water Power Density            | `vap_water_column_mean_sea_water_power_density`            | 0 - 1750  | W/m²             | 7               | cmocean.dense   |
| 95th Percentile Sea Water Power Density | `vap_water_column_95th_percentile_sea_water_power_density` | 0 - 32000 | W/m²             | 8               | cmocean.amp     |
| Mean Depth                              | `vap_sea_floor_depth`                                      | 0 - 200   | m (below NAVD88) | 10              | cmocean.deep    |

## Color Specifications

The following tables provide exact color specifications for each variable.
All colors use discrete levels with an overflow level for values exceeding the maximum range.

### Mean Sea Water Speed [m/s], `vap_water_column_mean_sea_water_speed`

- **Colormap:** cmocean.thermal
- **Data Range:** 0.0 to 1.5 m/s
- **Discrete Levels:** 11 (10 within range + 1 overflow level)

| Level | Value Range         | Hex Color | RGB Color            | Color Preview                                         |
| ----- | ------------------- | --------- | -------------------- | ----------------------------------------------------- |
| 1     | 0.00 - 0.15 \[m/s\] | `#032333` | `rgb(3, 35, 51)`     | ${\color[rgb]{0.012, 0.137, 0.200}\rule{30pt}{15pt}}$ |
| 2     | 0.15 - 0.30 \[m/s\] | `#0f3169` | `rgb(15, 49, 155)`   | ${\color[rgb]{0.059, 0.192, 0.412}\rule{30pt}{15pt}}$ |
| 3     | 0.30 - 0.45 \[m/s\] | `#3f339f` | `rgb(63, 51, 159)`   | ${\color[rgb]{0.247, 0.200, 0.624}\rule{30pt}{15pt}}$ |
| 4     | 0.45 - 0.60 \[m/s\] | `#674396` | `rgb(153, 67, 150)`  | ${\color[rgb]{0.404, 0.263, 0.588}\rule{30pt}{15pt}}$ |
| 5     | 0.60 - 0.75 \[m/s\] | `#8a528c` | `rgb(138, 82, 140)`  | ${\color[rgb]{0.541, 0.322, 0.549}\rule{30pt}{15pt}}$ |
| 6     | 0.75 - 0.90 \[m/s\] | `#b05f81` | `rgb(176, 95, 129)`  | ${\color[rgb]{0.690, 0.373, 0.506}\rule{30pt}{15pt}}$ |
| 7     | 0.90 - 1.05 \[m/s\] | `#d56b6c` | `rgb(213, 157, 158)` | ${\color[rgb]{0.835, 0.420, 0.424}\rule{30pt}{15pt}}$ |
| 8     | 1.05 - 1.20 \[m/s\] | `#f2824c` | `rgb(242, 130, 76)`  | ${\color[rgb]{0.949, 0.515, 0.298}\rule{30pt}{15pt}}$ |
| 9     | 1.20 - 1.35 \[m/s\] | `#fba53c` | `rgb(251, 165, 60)`  | ${\color[rgb]{0.984, 0.647, 0.235}\rule{30pt}{15pt}}$ |
| 15    | 1.35 - 1.50 \[m/s\] | `#f6d045` | `rgb(246, 208, 69)`  | ${\color[rgb]{0.965, 0.816, 0.271}\rule{30pt}{15pt}}$ |
| 11    | ≥ 1.500 m/s         | `#e7fa5a` | `rgb(231, 250, 90)`  | ${\color[rgb]{0.906, 0.980, 0.353}\rule{30pt}{15pt}}$ |

### 95th Percentile Sea Water Speed [m/s], `vap_water_column_95th_percentile_sea_water_speed`

- **Colormap:** cmocean.matter
- **Data Range:** 0.0 to 4.0 m/s
- **Discrete Levels:** 9 (8 within range + 1 overflow level)

| Level | Value Range       | Hex Color | RGB Color            | Color Preview                                                                                          |
| ----- | ----------------- | --------- | -------------------- | ------------------------------------------------------------------------------------------------------ |
| 1     | 0.00 - 0.50 [m/s] | `#fdedb0` | `rgb(253, 237, 176)` | <span style="background-color:#fdedb0; color:#fdedb0; padding:2px 8px; border-radius:3px;">████</span> |
| 2     | 0.50 - 1.00 [m/s] | `#f9c087` | `rgb(249, 192, 135)` | <span style="background-color:#f9c087; color:#f9c087; padding:2px 8px; border-radius:3px;">████</span> |
| 3     | 1.00 - 1.50 [m/s] | `#f19466` | `rgb(241, 148, 102)` | <span style="background-color:#f19466; color:#f19466; padding:2px 8px; border-radius:3px;">████</span> |
| 4     | 1.50 - 2.00 [m/s] | `#e56953` | `rgb(229, 105, 83)`  | <span style="background-color:#e56953; color:#e56953; padding:2px 8px; border-radius:3px;">████</span> |
| 5     | 2.00 - 2.50 [m/s] | `#ce4356` | `rgb(206, 67, 86)`   | <span style="background-color:#ce4356; color:#ce4356; padding:2px 8px; border-radius:3px;">████</span> |
| 6     | 2.50 - 3.00 [m/s] | `#ab2960` | `rgb(171, 41, 96)`   | <span style="background-color:#ab2960; color:#ab2960; padding:2px 8px; border-radius:3px;">████</span> |
| 7     | 3.00 - 3.50 [m/s] | `#821b62` | `rgb(130, 27, 98)`   | <span style="background-color:#821b62; color:#821b62; padding:2px 8px; border-radius:3px;">████</span> |
| 8     | 3.50 - 4.00 [m/s] | `#571656` | `rgb(87, 22, 86)`    | <span style="background-color:#571656; color:#571656; padding:2px 8px; border-radius:3px;">████</span> |
| 9     | ≥ 4.000 m/s       | `#2f0f3d` | `rgb(47, 15, 61)`    | <span style="background-color:#2f0f3d; color:#2f0f3d; padding:2px 8px; border-radius:3px;">████</span> |

### Mean Sea Water Power Density [W/m²], `vap_water_column_mean_sea_water_power_density`

- **Colormap:** cmocean.dense
- **Data Range:** 0 to 1750 W/m²
- **Discrete Levels:** 8 (7 within range + 1 overflow level)

| Level | Value Range         | Hex Color | RGB Color            | Color Preview                                                                                          |
| ----- | ------------------- | --------- | -------------------- | ------------------------------------------------------------------------------------------------------ |
| 1     | 0 - 250 [W/m^2]     | `#e6f0f0` | `rgb(230, 240, 240)` | <span style="background-color:#e6f0f0; color:#e6f0f0; padding:2px 8px; border-radius:3px;">████</span> |
| 2     | 250 - 500 [W/m^2]   | `#aad2e2` | `rgb(170, 210, 226)` | <span style="background-color:#aad2e2; color:#aad2e2; padding:2px 8px; border-radius:3px;">████</span> |
| 3     | 500 - 750 [W/m^2]   | `#7db0e3` | `rgb(125, 176, 227)` | <span style="background-color:#7db0e3; color:#7db0e3; padding:2px 8px; border-radius:3px;">████</span> |
| 4     | 750 - 1000 [W/m^2]  | `#7487e0` | `rgb(116, 135, 224)` | <span style="background-color:#7487e0; color:#7487e0; padding:2px 8px; border-radius:3px;">████</span> |
| 5     | 1000 - 1250 [W/m^2] | `#795cc3` | `rgb(121, 92, 195)`  | <span style="background-color:#795cc3; color:#795cc3; padding:2px 8px; border-radius:3px;">████</span> |
| 6     | 1250 - 1500 [W/m^2] | `#723693` | `rgb(114, 54, 147)`  | <span style="background-color:#723693; color:#723693; padding:2px 8px; border-radius:3px;">████</span> |
| 7     | 1500 - 1750 [W/m^2] | `#5c1957` | `rgb(92, 25, 87)`    | <span style="background-color:#5c1957; color:#5c1957; padding:2px 8px; border-radius:3px;">████</span> |
| 8     | ≥ 1750 W/m^2        | `#360e24` | `rgb(54, 14, 36)`    | <span style="background-color:#360e24; color:#360e24; padding:2px 8px; border-radius:3px;">████</span> |

### 95th Percentile Sea Water Power Density [W/m²], `vap_water_column_95th_percentile_sea_water_power_density`

- **Colormap:** cmocean.amp
- **Data Range:** 0 to 32000 W/m²
- **Discrete Levels:** 9 (8 within range + 1 overflow level)

| Level | Value Range           | Hex Color | RGB Color            | Color Preview                                                                                          |
| ----- | --------------------- | --------- | -------------------- | ------------------------------------------------------------------------------------------------------ |
| 1     | 0 - 4000 [W/m^2]      | `#f1ecec` | `rgb(241, 236, 236)` | <span style="background-color:#f1ecec; color:#f1ecec; padding:2px 8px; border-radius:3px;">████</span> |
| 2     | 4000 - 8000 [W/m^2]   | `#e2c7be` | `rgb(226, 199, 190)` | <span style="background-color:#e2c7be; color:#e2c7be; padding:2px 8px; border-radius:3px;">████</span> |
| 3     | 8000 - 12000 [W/m^2]  | `#d7a290` | `rgb(215, 162, 144)` | <span style="background-color:#d7a290; color:#d7a290; padding:2px 8px; border-radius:3px;">████</span> |
| 4     | 12000 - 16000 [W/m^2] | `#cc7d63` | `rgb(204, 125, 99)`  | <span style="background-color:#cc7d63; color:#cc7d63; padding:2px 8px; border-radius:3px;">████</span> |
| 5     | 16000 - 20000 [W/m^2] | `#bf583a` | `rgb(191, 88, 58)`   | <span style="background-color:#bf583a; color:#bf583a; padding:2px 8px; border-radius:3px;">████</span> |
| 6     | 20000 - 24000 [W/m^2] | `#ae2e24` | `rgb(174, 46, 36)`   | <span style="background-color:#ae2e24; color:#ae2e24; padding:2px 8px; border-radius:3px;">████</span> |
| 7     | 24000 - 28000 [W/m^2] | `#8e1028` | `rgb(142, 16, 40)`   | <span style="background-color:#8e1028; color:#8e1028; padding:2px 8px; border-radius:3px;">████</span> |
| 8     | 28000 - 32000 [W/m^2] | `#640e23` | `rgb(100, 14, 35)`   | <span style="background-color:#640e23; color:#640e23; padding:2px 8px; border-radius:3px;">████</span> |
| 9     | ≥ 32000 W/m^2         | `#3c0911` | `rgb(60, 9, 17)`     | <span style="background-color:#3c0911; color:#3c0911; padding:2px 8px; border-radius:3px;">████</span> |

### Mean Depth [m (below NAVD88)], `vap_sea_floor_depth`

- **Colormap:** cmocean.deep
- **Data Range:** 0 to 200 m (below NAVD88)
- **Discrete Levels:** 11 (10 within range + 1 overflow level)

| Level | Value Range       | Hex Color | RGB Color            | Color Preview                                                                                          |
| ----- | ----------------- | --------- | -------------------- | ------------------------------------------------------------------------------------------------------ |
| 1     | 0.00 - 20.00 [m]  | `#fdfdcc` | `rgb(253, 253, 204)` | <span style="background-color:#fdfdcc; color:#fdfdcc; padding:2px 8px; border-radius:3px;">████</span> |
| 2     | 20.00 - 40.00 [m] | `#c9ebb1` | `rgb(201, 235, 177)` | <span style="background-color:#c9ebb1; color:#c9ebb1; padding:2px 8px; border-radius:3px;">████</span> |
| 3     | 40.00 - 60.00 [m] | `#91d8a3` | `rgb(145, 216, 163)` | <span style="background-color:#91d8a3; color:#91d8a3; padding:2px 8px; border-radius:3px;">████</span> |
| 4     | 60.00 - 80.00 [m] | `#66c2a3` | `rgb(102, 194, 163)` | <span style="background-color:#66c2a3; color:#66c2a3; padding:2px 8px; border-radius:3px;">████</span> |
| 5     | 80 - 100 [m]      | `#51a8a2` | `rgb(81, 168, 162)`  | <span style="background-color:#51a8a2; color:#51a8a2; padding:2px 8px; border-radius:3px;">████</span> |
| 6     | 100 - 120 [m]     | `#488d9d` | `rgb(72, 141, 157)`  | <span style="background-color:#488d9d; color:#488d9d; padding:2px 8px; border-radius:3px;">████</span> |
| 7     | 120 - 140 [m]     | `#407598` | `rgb(64, 117, 152)`  | <span style="background-color:#407598; color:#407598; padding:2px 8px; border-radius:3px;">████</span> |
| 8     | 140 - 160 [m]     | `#3d5a92` | `rgb(61, 90, 146)`   | <span style="background-color:#3d5a92; color:#3d5a92; padding:2px 8px; border-radius:3px;">████</span> |
| 9     | 160 - 180 [m]     | `#41407b` | `rgb(65, 64, 123)`   | <span style="background-color:#41407b; color:#41407b; padding:2px 8px; border-radius:3px;">████</span> |
| 10    | 180 - 200 [m]     | `#372c50` | `rgb(55, 44, 80)`    | <span style="background-color:#372c50; color:#372c50; padding:2px 8px; border-radius:3px;">████</span> |
| 11    | ≥ 200.0 m         | `#271a2c` | `rgb(39, 26, 44)`    | <span style="background-color:#271a2c; color:#271a2c; padding:2px 8px; border-radius:3px;">████</span> |

## Visualizations by Variable

### Mean Sea Water Speed

**Puget Sound, Washington Mean Sea Water Speed**

![Mean Sea Water Speed for Puget Sound, Washington](docs/img/WA_puget_sound_mean_sea_water_speed.png)
\*Figure: Mean Sea Water Speed spatial distribution for Puget Sound, Washington. Units: m/s

**Piscataqua River, New Hampshire Mean Sea Water Speed**

![Mean Sea Water Speed for Piscataqua River, New Hampshire](docs/img/NH_piscataqua_river_mean_sea_water_speed.png)
\*Figure: Mean Sea Water Speed spatial distribution for Piscataqua River, New Hampshire. Units: m/s

**Western Passage, Maine Mean Sea Water Speed**

![Mean Sea Water Speed for Western Passage, Maine](docs/img/ME_western_passage_mean_sea_water_speed.png)
\*Figure: Mean Sea Water Speed spatial distribution for Western Passage, Maine. Units: m/s

**Cook Inlet, Alaska Mean Sea Water Speed**

![Mean Sea Water Speed for Cook Inlet, Alaska](docs/img/AK_cook_inlet_mean_sea_water_speed.png)
\*Figure: Mean Sea Water Speed spatial distribution for Cook Inlet, Alaska. Units: m/s

**Aleutian Islands, Alaska Mean Sea Water Speed**

![Mean Sea Water Speed for Aleutian Islands, Alaska](docs/img/AK_aleutian_islands_mean_sea_water_speed.png)
\*Figure: Mean Sea Water Speed spatial distribution for Aleutian Islands, Alaska. Units: m/s

---

### 95th Percentile Sea Water Speed

**Puget Sound, Washington 95th Percentile Sea Water Speed**

![95th Percentile Sea Water Speed for Puget Sound, Washington](docs/img/WA_puget_sound_p95_sea_water_speed.png)
\*Figure: 95th Percentile Sea Water Speed spatial distribution for Puget Sound, Washington. Units: m/s

**Piscataqua River, New Hampshire 95th Percentile Sea Water Speed**

![95th Percentile Sea Water Speed for Piscataqua River, New Hampshire](docs/img/NH_piscataqua_river_p95_sea_water_speed.png)
\*Figure: 95th Percentile Sea Water Speed spatial distribution for Piscataqua River, New Hampshire. Units: m/s

**Western Passage, Maine 95th Percentile Sea Water Speed**

![95th Percentile Sea Water Speed for Western Passage, Maine](docs/img/ME_western_passage_p95_sea_water_speed.png)
\*Figure: 95th Percentile Sea Water Speed spatial distribution for Western Passage, Maine. Units: m/s

**Cook Inlet, Alaska 95th Percentile Sea Water Speed**

![95th Percentile Sea Water Speed for Cook Inlet, Alaska](docs/img/AK_cook_inlet_p95_sea_water_speed.png)
\*Figure: 95th Percentile Sea Water Speed spatial distribution for Cook Inlet, Alaska. Units: m/s

**Aleutian Islands, Alaska 95th Percentile Sea Water Speed**

![95th Percentile Sea Water Speed for Aleutian Islands, Alaska](docs/img/AK_aleutian_islands_p95_sea_water_speed.png)
\*Figure: 95th Percentile Sea Water Speed spatial distribution for Aleutian Islands, Alaska. Units: m/s

---

### Mean Sea Water Power Density

**Puget Sound, Washington Mean Sea Water Power Density**

![Mean Sea Water Power Density for Puget Sound, Washington](docs/img/WA_puget_sound_mean_sea_water_power_density.png)
\*Figure: Mean Sea Water Power Density spatial distribution for Puget Sound, Washington. Units: m/s

**Piscataqua River, New Hampshire Mean Sea Water Power Density**

![Mean Sea Water Power Density for Piscataqua River, New Hampshire](docs/img/NH_piscataqua_river_mean_sea_water_power_density.png)
\*Figure: Mean Sea Water Power Density spatial distribution for Piscataqua River, New Hampshire. Units: m/s

**Western Passage, Maine Mean Sea Water Power Density**

![Mean Sea Water Power Density for Western Passage, Maine](docs/img/ME_western_passage_mean_sea_water_power_density.png)
\*Figure: Mean Sea Water Power Density spatial distribution for Western Passage, Maine. Units: m/s

**Cook Inlet, Alaska Mean Sea Water Power Density**

![Mean Sea Water Power Density for Cook Inlet, Alaska](docs/img/AK_cook_inlet_mean_sea_water_power_density.png)
\*Figure: Mean Sea Water Power Density spatial distribution for Cook Inlet, Alaska. Units: m/s

**Aleutian Islands, Alaska Mean Sea Water Power Density**

![Mean Sea Water Power Density for Aleutian Islands, Alaska](docs/img/AK_aleutian_islands_mean_sea_water_power_density.png)
\*Figure: Mean Sea Water Power Density spatial distribution for Aleutian Islands, Alaska. Units: m/s

---

### 95th Percentile Sea Water Power Density

**Puget Sound, Washington 95th Percentile Sea Water Power Density**

![95th Percentile Sea Water Power Density for Puget Sound, Washington](docs/img/WA_puget_sound_p95_sea_water_power_density.png)
\*Figure: 95th Percentile Sea Water Power Density spatial distribution for Puget Sound, Washington. Units: m/s

**Piscataqua River, New Hampshire 95th Percentile Sea Water Power Density**

![95th Percentile Sea Water Power Density for Piscataqua River, New Hampshire](docs/img/NH_piscataqua_river_p95_sea_water_power_density.png)
\*Figure: 95th Percentile Sea Water Power Density spatial distribution for Piscataqua River, New Hampshire. Units: m/s

**Western Passage, Maine 95th Percentile Sea Water Power Density**

![95th Percentile Sea Water Power Density for Western Passage, Maine](docs/img/ME_western_passage_p95_sea_water_power_density.png)
\*Figure: 95th Percentile Sea Water Power Density spatial distribution for Western Passage, Maine. Units: m/s

**Cook Inlet, Alaska 95th Percentile Sea Water Power Density**

![95th Percentile Sea Water Power Density for Cook Inlet, Alaska](docs/img/AK_cook_inlet_p95_sea_water_power_density.png)
\*Figure: 95th Percentile Sea Water Power Density spatial distribution for Cook Inlet, Alaska. Units: m/s

**Aleutian Islands, Alaska 95th Percentile Sea Water Power Density**

![95th Percentile Sea Water Power Density for Aleutian Islands, Alaska](docs/img/AK_aleutian_islands_p95_sea_water_power_density.png)
\*Figure: 95th Percentile Sea Water Power Density spatial distribution for Aleutian Islands, Alaska. Units: m/s

---

### Distance to Sea Floor

**Puget Sound, Washington Distance to Sea Floor**

![Distance to Sea Floor for Puget Sound, Washington](docs/img/WA_puget_sound_distance_to_sea_floor.png)
\*Figure: Distance to Sea Floor spatial distribution for Puget Sound, Washington. Units:

**Piscataqua River, New Hampshire Distance to Sea Floor**

![Distance to Sea Floor for Piscataqua River, New Hampshire](docs/img/NH_piscataqua_river_distance_to_sea_floor.png)
\*Figure: Distance to Sea Floor spatial distribution for Piscataqua River, New Hampshire. Units:

**Western Passage, Maine Distance to Sea Floor**

![Distance to Sea Floor for Western Passage, Maine](docs/img/ME_western_passage_distance_to_sea_floor.png)
\*Figure: Distance to Sea Floor spatial distribution for Western Passage, Maine. Units:

**Cook Inlet, Alaska Distance to Sea Floor**

![Distance to Sea Floor for Cook Inlet, Alaska](docs/img/AK_cook_inlet_distance_to_sea_floor.png)
\*Figure: Distance to Sea Floor spatial distribution for Cook Inlet, Alaska. Units:

**Aleutian Islands, Alaska Distance to Sea Floor**

![Distance to Sea Floor for Aleutian Islands, Alaska](docs/img/AK_aleutian_islands_distance_to_sea_floor.png)
\*Figure: Distance to Sea Floor spatial distribution for Aleutian Islands, Alaska. Units:

---

## Cross-Regional Comparative Analysis

Comparative visualizations across all processed regions provide insights into spatial variability, statistical patterns, and visualization parameter validation.

### Visualization Maximum Justification

These comprehensive plots validate the chosen visualization maximum (viz_max) parameters used throughout the analysis. Each visualization demonstrates that the selected cutoff values effectively capture the bulk of the data while filtering extreme outliers, ensuring meaningful and readable visualizations.

**Mean Sea Water Speed - Visualization Maximum Validation**

![Mean Sea Water Speed Viz Max Justification](docs/img/vap_water_column_mean_sea_water_speed_viz_max_justification.png)
_Figure: Comprehensive validation of visualization maximum for mean sea water speed. Shows full data distribution, regional comparisons within bounds, key statistics, and outlier assessment. Units: m/s. Validates the visualization maximum used for mean sea water speed analysis, showing data retention rates and outlier filtering effectiveness._

**95th Percentile Sea Water Speed - Visualization Maximum Validation**

![95th Percentile Sea Water Speed Viz Max Justification](docs/img/vap_water_column_95th_percentile_sea_water_speed_viz_max_justification.png)
_Figure: Comprehensive validation of visualization maximum for 95th percentile sea water speed. Shows full data distribution, regional comparisons within bounds, key statistics, and outlier assessment. Units: m/s. Demonstrates the appropriateness of the visualization cutoff for 95th percentile sea water speed values across all regions._

**Mean Sea Water Power Density - Visualization Maximum Validation**

![Mean Sea Water Power Density Viz Max Justification](docs/img/vap_water_column_mean_sea_water_power_density_viz_max_justification.png)
_Figure: Comprehensive validation of visualization maximum for mean sea water power density. Shows full data distribution, regional comparisons within bounds, key statistics, and outlier assessment. Units: W/m². Justifies the power density visualization maximum by showing statistical distribution and outlier characteristics._

**95th Percentile Sea Water Power Density - Visualization Maximum Validation**

![95th Percentile Sea Water Power Density Viz Max Justification](docs/img/vap_water_column_95th_percentile_sea_water_power_density_viz_max_justification.png)
_Figure: Comprehensive validation of visualization maximum for 95th percentile sea water power density. Shows full data distribution, regional comparisons within bounds, key statistics, and outlier assessment. Units: W/m². Validates the visualization bounds for 95th percentile power density measurements across regional datasets._

**Sea Floor Depth - Visualization Maximum Validation**

![Sea Floor Depth Viz Max Justification](docs/img/vap_sea_floor_depth_viz_max_justification.png)
_Figure: Comprehensive validation of visualization maximum for sea floor depth. Shows full data distribution, regional comparisons within bounds, key statistics, and outlier assessment. Units: m. Shows the effectiveness of depth visualization parameters in capturing bathymetric variability while controlling for extreme outliers._

### Regional Distribution Comparisons

These kernel density estimation (KDE) plots provide clean statistical comparisons of variable distributions across all processed regions, focused within the validated visualization ranges.

**Mean Sea Water Speed Distribution Comparison**

![Mean Sea Water Speed Regional Comparison](docs/img/vap_water_column_mean_sea_water_speed_regional_comparison.png)
_Figure: Kernel density estimation comparison of mean sea water speed across all processed regions. Units: m/s. Regional distribution patterns for mean sea water speed. Distributions are shown within validated visualization bounds for optimal clarity._

**95th Percentile Sea Water Speed Distribution Comparison**

![95th Percentile Sea Water Speed Regional Comparison](docs/img/vap_water_column_95th_percentile_sea_water_speed_regional_comparison.png)
_Figure: Kernel density estimation comparison of 95th percentile sea water speed across all processed regions. Units: m/s. Comparative analysis of high-speed current characteristics across regions. Distributions are shown within validated visualization bounds for optimal clarity._

**Mean Sea Water Power Density Distribution Comparison**

![Mean Sea Water Power Density Regional Comparison](docs/img/vap_water_column_mean_sea_water_power_density_regional_comparison.png)
_Figure: Kernel density estimation comparison of mean sea water power density across all processed regions. Units: W/m². Power density distribution comparison highlighting regional resource potential. Distributions are shown within validated visualization bounds for optimal clarity._

**95th Percentile Sea Water Power Density Distribution Comparison**

![95th Percentile Sea Water Power Density Regional Comparison](docs/img/vap_water_column_95th_percentile_sea_water_power_density_regional_comparison.png)
_Figure: Kernel density estimation comparison of 95th percentile sea water power density across all processed regions. Units: W/m². High-power density event comparison across different oceanic regions. Distributions are shown within validated visualization bounds for optimal clarity._

**Sea Floor Depth Distribution Comparison**

![Sea Floor Depth Regional Comparison](docs/img/vap_sea_floor_depth_regional_comparison.png)
_Figure: Kernel density estimation comparison of sea floor depth across all processed regions. Units: m. Bathymetric distribution comparison showing depth characteristics by region. Distributions are shown within validated visualization bounds for optimal clarity._

### Percentile Bar Chart Comparisons

These charts provide quantitative comparison of key percentile values across regions, with visualization maximum reference lines for context.

**Mean Sea Water Speed Percentile Comparison**

![Mean Sea Water Speed Bar Comparison](docs/img/vap_water_column_mean_sea_water_speed_bar_comparison.png)
_Figure: Percentile values of mean sea water speed compared across all processed regions. Units: m/s. Quantitative percentile comparison with visualization bounds overlay. Enables quantitative assessment of regional variability and extreme value characteristics._

**95th Percentile Sea Water Speed Percentile Comparison**

![95th Percentile Sea Water Speed Bar Comparison](docs/img/vap_water_column_95th_percentile_sea_water_speed_bar_comparison.png)
_Figure: Percentile values of 95th percentile sea water speed compared across all processed regions. Units: m/s. High-speed percentile values across regions with reference thresholds. Enables quantitative assessment of regional variability and extreme value characteristics._

**Mean Sea Water Power Density Percentile Comparison**

![Mean Sea Water Power Density Bar Comparison](docs/img/vap_water_column_mean_sea_water_power_density_bar_comparison.png)
_Figure: Percentile values of mean sea water power density compared across all processed regions. Units: W/m². Power density percentile analysis with visualization maximum context. Enables quantitative assessment of regional variability and extreme value characteristics._

**95th Percentile Sea Water Power Density Percentile Comparison**

![95th Percentile Sea Water Power Density Bar Comparison](docs/img/vap_water_column_95th_percentile_sea_water_power_density_bar_comparison.png)
_Figure: Percentile values of 95th percentile sea water power density compared across all processed regions. Units: W/m². Regional power density extremes with validated cutoff references. Enables quantitative assessment of regional variability and extreme value characteristics._

**Sea Floor Depth Percentile Comparison**

![Sea Floor Depth Bar Comparison](docs/img/vap_sea_floor_depth_bar_comparison.png)
_Figure: Percentile values of sea floor depth compared across all processed regions. Units: m. Depth percentile comparison across bathymetric regions. Enables quantitative assessment of regional variability and extreme value characteristics._

### Visualization Methodology Notes

**Visualization Maximum (Viz Max) Approach**: All visualizations use validated maximum values that capture 95-99.9% of the data while filtering extreme outliers. This approach ensures:

- Clear, readable visualizations without distortion from extreme values
- Consistent scales across regional comparisons
- Transparent documentation of data filtering decisions
- Preservation of statistical integrity for the bulk of the dataset

**Data Retention**: The justification plots show exactly what percentage of data is retained vs. filtered, providing full transparency about the visualization choices and their impact on the analysis.

---

## Document Information

- **Generated:** 2025-06-03 12:40:07 UTC
- **Regions Processed:** WA_puget_sound, NH_piscataqua_river, ME_western_passage, AK_cook_inlet, AK_aleutian_islands

_This specification was auto-generated from the tidal data visualization pipeline._
_All color codes, ranges, and technical specifications are programmatically derived._
