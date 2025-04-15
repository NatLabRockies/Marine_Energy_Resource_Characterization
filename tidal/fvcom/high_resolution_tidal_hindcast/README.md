# WPTO High Resolution Tidal Hindcast

2025-04-15

- [<span class="toc-section-number">1</span> Overview](#overview)
- [<span class="toc-section-number">2</span> Versions](#versions)
- [<span class="toc-section-number">3</span> Locations](#locations)
- [<span class="toc-section-number">4</span> HPC Data
  Locations](#hpc-data-locations)
- [<span class="toc-section-number">5</span> Model Data
  Specification](#model-data-specification)
- [<span class="toc-section-number">6</span> Data Quality Assurance and
  Quality Control](#data-quality-assurance-and-quality-control)
- [<span class="toc-section-number">7</span> Derived Variables, Value
  Added Products (VAP)](#derived-variables-value-added-products-vap)
- [<span class="toc-section-number">8</span> Data Quality Assurance and
  Quality Control](#data-quality-assurance-and-quality-control-1)
- [<span class="toc-section-number">9</span> Data Levels and Processing
  Pipeline](#data-levels-and-processing-pipeline)
- [<span class="toc-section-number">10</span> Running Standardization
  Code](#running-standardization-code)
- [<span class="toc-section-number">11</span> Included
  Metadata](#included-metadata)
- [<span class="toc-section-number">12</span> Visualization
  Specification](#visualization-specification)
- [<span class="toc-section-number">13</span>
  Acknowledgement](#acknowledgement)
- [<span class="toc-section-number">14</span> Citation](#citation)
- [<span class="toc-section-number">15</span> References](#references)

# Overview

This repository contains the code and methodology for processing and
visualizing high-resolution tidal hindcast data generated using the
Finite Volume Community Ocean Model (FVCOM) at five strategically
selected U.S. coastal locations. The project represents a collaborative
effort between Pacific Northwest National Laboratory (data generation)
and National Renewable Energy Laboratory (data processing and
visualization), to generate original high resolution tidal data then
standardize and summarize the outputs into accessible resource data for
marine energy assessment. The processing pipeline in this repository
converts approximately 10 terabytes of raw model data through a series
of quality-controlled stages, standardizing coordinates to consistent
geographic reference systems, calculating key variables such as water
speed and power density, and generating depth averaged variables and
yearly averaged summaries. Complete and summarized standardized datasets
can be be found on OpenEI, and summary data is visualized on the Marine
Energy Atlas

## High Resolution Tidal Hindcast Overview

<div id="tbl-hrth-overview">

| Label | Specification |
|----|----|
| Project Title | High Resolution Tidal Hindcast |
| Project ID | tidal_hindcast_fvcom |
| Data Generation Organization | Pacific Northwest National Laboratory (PNNL) |
| Data Processing and Visualization Organization | National Renewable Energy Laboratory (NREL) |
| Brief Description | 1 year of high resolution tidal data at 5 targeted US locations, generated using the Finite Volume Community Ocean Model (FVCOM) |

Table 1: High Resolution Tidal Hindcast Overview

</div>

## Program Links

<https://www.energy.gov/eere/water/marine-energy-resource-assessment-and-characterization>
<https://www.pnnl.gov/marine-energy-resource-characterization>
<https://www.nrel.gov/water/resource-characterization>

## Processing Software Links

<https://github.com/NREL/Marine_Energy_Resource_Characterization>

## Available Data Overview

5 US locations of new high resolution 3D tidal data, generated using
FVCOM version 4.3.1, with `u` and `v` vectors, and calculated
`sea_water_speed`, `sea_water_to_direction`, and
`sea_water_power_density` at 10 sigma layers:

<div id="fig-ak-cook-speed">

![](./docs/img/ak_cook_inlet_speed.png)

Figure 1: Cook Inlet, Alaska - Sea Water Speed, Yearly Average, Depth
Average

</div>

<div id="fig-ak-aleutian-speed">

![](./docs/img/ak_aleutian_islands_speed.png)

Figure 2: Aleutian Islands - Sea Water Speed, Yearly Average, Depth
Average

</div>

<div id="fig-me-west-speed-speed">

![](./docs/img/ak_cook_inlet_speed.png)

Figure 3: Western Passage, Maine - Sea Water Speed, Yearly Average,
Depth Average

</div>

<div id="fig-nh-piscataqua-speed">

![](./docs/img/nh_piscataqua_river_speed.png)

Figure 4: Piscataqua River New Hampshire - Sea Water Speed, Yearly
Average, Depth Average

</div>

<div id="fig-wa-puget-speed">

![](./docs/img/wa_puget_sound_speed.png)

Figure 5: Puget Sound, Washington - Sea Water Speed, Yearly Average,
Depth Average

</div>

## Data Output Plan

| Data Level | Description | Format | Storage Location | Public Access |
|----|----|----|----|----|
| `00_raw` | Original NetCDF files from FVCOM model | NetCDF (nc) | HPC Kestrel / AWS Archive | No |
| `a1_std` | Standardized data with consistent naming and attributes | NetCDF (nc) | HPC Kestrel | No |
| `a2_std_partition` | Standardized data partitioned by time chunks for processing | NetCDF (nc) | HPC Kestrel | No |
| `b1_vap` | Value-added products with derived variables, full temporal resolution | NetCDF (nc) | HPC Kestrel / OpenEI AWS | Yes |
| `b2_summary_vap` | Summary statistics of value-added products | NetCDF (nc) | HPC Kestrel / OpenEI AWS | Yes |
| `b3_vap_partition` | Partitioned VAP data for efficient processing | NetCDF (nc) | HPC Kestrel / OpenEI AWS | Yes |
| `b4_vap_summary_parquet` | Summary data in Parquet format for analytics | Parquet | HPC Kestrel / OpenEI AWS | Yes |
| `b5_vap_atlas_summary_parquet` | Atlas-ready summary data for visualization | Parquet | HPC Kestrel / OpenEI AWS | Yes |

# Versions

| Asset                             | Version     |
|-----------------------------------|-------------|
| Model                             | FVCOM_4.3.1 |
| Processing Code (This Repository) | 0.2.0       |
| Standardized Dataset              | 0.2.0       |

## Model Configuration

<div id="tbl-model-config">

| Model Configuration | Specification |
|----|----|
| Model | FVCOM, https://www.fvcom.org/ |
| Documentation Link | [FVCOM Publicly Available Manual - 3.1.6](https://etchellsfleet27.com/wp-content/uploads/2020/06/FVCOM_User_Manual_v3.1.6.pdf) |
| Dimensions | 3D, over time |
| Horizontal Resolution | grid node and element points |
| Vertical Resolution | 10 uniform sigma-stretched coordinate |
| Horizontal Coordinates | lat and lon, UTM zone, State Plane |
| Vertical Datum | mean sea level (MSL), NAVD88 |
| Temporal Resolution | hourly, half-hourly |
| Wetting & drying feature | activated |
| Boundary forcing | 12 tidal constituents from OSU TPXO Tide Models |
| Wind | ERA5 or CFSv2 |

Table 2: High Resolution Tidal Hindcast Model Configuration per PNNL

</div>

## Standards

<div id="tbl-iec-compliance">

| IEC Standards (IEC TS 62600-201, 2015) | Compliance Status |
|----|----|
| Stage 1 (feasibility) tidal resource assessment requirements | Meet all requirements except those listed below |
| Wave-current interaction | Not considered because wave is small in the study domain |
| Atmospheric forcing (Wind and pressure) | Not considered because effects of wind and atmospheric pressure on tidal currents in the domain is negligible |
| Seawater density, salinity and temperature | Not considered because density-induced estuarine flow is small in the domain |

Table 3: High Resolution Tidal Hindcast IEC Standards Compliance per
PNNL

</div>

# Locations

<div id="tbl-loc-spec">

<div class="cell-output cell-output-display cell-output-markdown">

| Location Name | Output Name | Input Directory |
|:---|:---|:---|
| Aleutian Islands, Alaska | AK_aleutian_islands | Aleutian_Islands_year |
| Cook Inlet, Alaska | AK_cook_inlet | Cook_Inlet_PNNL |
| Piscataqua River, New Hampshire | NH_piscataqua_river | PIR_full_year |
| Puget Sound, Washington | WA_puget_sound | Puget_Sound_corrected |
| Western Passage, Maine | ME_western_passage | Western_Passage_corrected |

</div>

Table 4: High Resolution Tidal Hindcast Available Locations

</div>

## Time Specification

<div id="tbl-loc-spec-time">

<div class="cell-output cell-output-display cell-output-markdown">

| Location Name | Data Start Time \[UTC\] | Data End Time \[UTC\] | Sampling Frequency | Time Count |
|:---|:---|:---|:---|---:|
| Aleutian Islands, Alaska | 2010-06-03 00:00:00 | 2011-06-02 23:00:00 | Hourly | 8760 |
| Cook Inlet, Alaska | 2005-01-01 00:00:00 | 2005-12-31 23:00:00 | Hourly | 8760 |
| Piscataqua River, New Hampshire | 2007-01-01 00:00:00 | 2007-12-31 23:30:00 | Half-Hourly | 17520 |
| Puget Sound, Washington | 2015-01-01 00:00:00 | 2015-12-30 23:30:00 | Half-Hourly | 17472 |
| Western Passage, Maine | 2017-01-01 00:00:00 | 2017-12-31 23:30:00 | Half-Hourly | 17520 |

</div>

Table 5: High Resolution Tidal Hindcast Time Specification

</div>

## Data Format

| Label                       | Specification   |
|-----------------------------|-----------------|
| File Format(s)              | NetCDF4, `*.nc` |
| Total Combined Dataset Size | ~10TB           |

## Original Data Volume Format

| Location | File Count | Directory Count | Total Data Volume | Average File Size | Duration Per File |
|----|----|----|----|----|----|
| AK, Aleutian Islands | 370 | 1 | 2.72 TB | 6.9 GB | 1 Day |
| AK, Cook Inlet | 366 | 1 | 1.31 TB | 3.4 GB | 1 Day |
| ME, Western Passage | 406 | 6 | 2.53 TB | 33 GB | 5 Days |
| NH, Piscataqua River | 370 | 1 | 1.60 TB | 4.1 GB | 1 Day |
| WA, Puget Sound | 83 | 13 | 3.33 TB | 45 GB | 5 Days |

## Temporal Details

| Label | Specification |
|----|----|
| Date/Time Format | modified julian day (MJD), `time`, UTC String, `Times` |
| Timezone | UTC |
| Expected Start Date | Varies by location (see below) |
| Expected End Date | Varies by location (see below) |
| Sampling Frequency, $\Delta t$ \[s\] | 1800s (30min) or 3600s (60min) by location |

### Location-Specific Temporal Details

| Location | Start Date | End Date | Temporal Resolution | $\Delta t$ \[s\] |
|----|----|----|----|----|
| Aleutian Islands | 2010-06-03 00:00:00 | 2011-06-02 23:00:00 | hourly | 3600 |
| Cook Inlet | 2005-01-01 00:00:00 | 2005-12-31 23:00:00 | hourly | 3600 |
| Western Passage | 2017-01-01 00:00:00 | 2017-12-31 23:30:00 | half-hourly | 1800 |
| Piscataqua River | 2007-01-01 00:00:00 | 2007-12-31 23:30:00 | half-hourly | 1800 |
| Puget Sound | 2015-01-01 00:00:00 | 2015-12-30 23:30:00 | half-hourly | 1800 |

## Spatial Details

| Label | Specification |
|----|----|
| Location Names | AK Aleutian Islands, AK Cook Inlet, ME Western Passage, NH Piscataqua River, WA Puget Sound |
| Coordinate Format | latitude/longitude or UTM (see below) |
| Grid Type | Unstructured triangular grid (FVCOM) |
| Grid Dimensions | Varies by location (nodes, elements/faces) |
| Grid Dimension Details | Nodes at vertices, elements/faces at centers |
| Spatial Uncertainty | Not Defined |

### Location-Specific Spatial Details

| Location         | Coordinate System  | Notes                    |
|------------------|--------------------|--------------------------|
| Aleutian Islands | latitude/longitude | Global coordinate system |
| Cook Inlet       | latitude/longitude | Global coordinate system |
| Western Passage  | UTM zone 19        | Local projection         |
| Piscataqua River | UTM zone 19        | Local projection         |
| Puget Sound      | UTM zone 10        | Local projection         |

## Dimension Details

| Label | Specification |
|----|----|
| Required Dimensions | time, node, nele/face, siglay/sigma_layer |
| Required Dimension Data Types | time: float32/datetime64, node: int64, nele/face: int64, siglay: float32 |
| Required Dimension Descriptions | time: temporal dimension, node: mesh vertices, nele/face: mesh elements, siglay: vertical sigma layers |
| Optional Dimensions | siglev, three, face_node_index |
| Optional Dimension Data Types | siglev: float32, three: int64, face_node_index: int64 |
| Optional Dimension Descriptions | siglev: sigma level interfaces, three: triangle connectivity, face_node_index: mapping between faces and nodes |

# HPC Data Locations

<div id="tbl-test">

<div class="cell-output cell-output-display cell-output-markdown">

| Type | Data Level | Name | Kestrel Path |
|:---|:---|:---|:---|
| Input | 00 | Original | `/projects/hindcastra/Tidal/datasets/high_resolution_tidal_hindcast/<location>/00_raw` |
| Output | A1 | Standardized | `/projects/hindcastra/Tidal/datasets/high_resolution_tidal_hindcast/<location>/a1_std` |
| Output | A2 | Standardized Partition | `/projects/hindcastra/Tidal/datasets/high_resolution_tidal_hindcast/<location>/a2_std_partition` |
| Output | B1 | Vap | `/projects/hindcastra/Tidal/datasets/high_resolution_tidal_hindcast/<location>/b1_vap` |
| Output | B2 | Summary Vap | `/projects/hindcastra/Tidal/datasets/high_resolution_tidal_hindcast/<location>/b2_summary_vap` |
| Output | B3 | Vap Partition | `/projects/hindcastra/Tidal/datasets/high_resolution_tidal_hindcast/<location>/b3_vap_partition` |
| Output | B4 | Vap Summary Parquet | `/projects/hindcastra/Tidal/datasets/high_resolution_tidal_hindcast/<location>/b4_vap_summary_parquet` |
| Output | B5 | Vap Atlas Summary Parquet | `/projects/hindcastra/Tidal/datasets/high_resolution_tidal_hindcast/<location>/b5_vap_atlas_summary_parquet` |
| Output | Z99 | Tracking | `/projects/hindcastra/Tidal/datasets/high_resolution_tidal_hindcast/<location>/z99_tracking` |

</div>

Table 6: Available Data on Kestrel

</div>

# Model Data Specification

## Model Version Numbers

| Name          | Specification |
|---------------|---------------|
| Model Version | FVCOM_4.3.1   |
| Conventions   | CF-1.0        |

## Required Original Variables

These variables with the specified types must exist in the input
original datasets

<div id="tbl-req-orig-vrs">

<div class="cell-output cell-output-display cell-output-markdown">

| Variable | Data Type | Coordinates | Dimensions |
|:---|:---|:---|:---|
| time | float32 | \[‘time’\] | \[‘time’\] |
| Times | \|S26 | \[‘time’\] | \[‘time’\] |
| lat | float32 | \[‘lon’, ‘lat’\] | \[‘node’\] |
| lon | float32 | \[‘lon’, ‘lat’\] | \[‘node’\] |
| latc | float32 | \[‘lonc’, ‘latc’\] | \[‘nele’\] |
| lonc | float32 | \[‘lonc’, ‘latc’\] | \[‘nele’\] |
| x | float32 | \[‘lon’, ‘lat’\] | \[‘node’\] |
| y | float32 | \[‘lon’, ‘lat’\] | \[‘node’\] |
| xc | float32 | \[‘lonc’, ‘latc’\] | \[‘nele’\] |
| yc | float32 | \[‘lonc’, ‘latc’\] | \[‘nele’\] |
| nele | int64 | \[‘lonc’, ‘latc’\] | \[‘nele’\] |
| node | int64 | \[‘lon’, ‘lat’\] | \[‘node’\] |
| nv | int32 | \[‘lonc’, ‘latc’\] | \[‘three’, ‘nele’\] |
| three | int64 | \[\] | \[‘three’\] |
| zeta | float32 | \[‘lon’, ‘lat’, ‘time’\] | \[‘time’, ‘node’\] |
| h_center | float32 | \[‘lonc’, ‘latc’\] | \[‘nele’\] |
| siglev_center | float32 | \[‘lonc’, ‘latc’\] | \[‘siglev’, ‘nele’\] |
| u | float32 | \[‘lonc’, ‘latc’, ‘time’\] | \[‘time’, ‘siglay’, ‘nele’\] |
| v | float32 | \[‘lonc’, ‘latc’, ‘time’\] | \[‘time’, ‘siglay’, ‘nele’\] |

</div>

Table 7: Required Original Variables

</div>

<div id="tbl-req-orig-vrs-2">

<div class="cell-output cell-output-display cell-output-markdown">

| Variable | Long Name | Standard Name | Units | Format | Time Zone |
|:---|:---|:---|:---|:---|:---|
| time | time | Not Found | days since 1858-11-17 00:00:00 | modified julian day (MJD) | UTC |
| Times | Not Found | Not Found |  | N/A | UTC |
| lat | nodal latitude | latitude | degrees_north | N/A | N/A |
| lon | nodal longitude | longitude | degrees_east | N/A | N/A |
| latc | zonal latitude | latitude | degrees_north | N/A | N/A |
| lonc | zonal longitude | longitude | degrees_east | N/A | N/A |
| x | nodal x-coordinate | Not Found | meters | N/A | N/A |
| y | nodal y-coordinate | Not Found | meters | N/A | N/A |
| xc | zonal x-coordinate | Not Found | meters | N/A | N/A |
| yc | zonal y-coordinate | Not Found | meters | N/A | N/A |
| nele | Not Found | Not Found |  | N/A | N/A |
| node | Not Found | Not Found |  | N/A | N/A |
| nv | nodes surrounding element | Not Found |  | N/A | N/A |
| three | Not Found | Not Found |  | N/A | N/A |
| zeta | Water Surface Elevation | sea_surface_height_above_geoid | meters | N/A | N/A |
| h_center | Bathymetry | sea_floor_depth_below_geoid | m | N/A | N/A |
| siglev_center | Sigma Levels | ocean_sigma/general_coordinate |  | N/A | N/A |
| u | Eastward Water Velocity | eastward_sea_water_velocity | meters s-1 | N/A | N/A |
| v | Northward Water Velocity | Northward_sea_water_velocity | meters s-1 | N/A | N/A |

</div>

Table 8: Required Original Variables, Continued

</div>

# Data Quality Assurance and Quality Control

Data quality assurance and quality control (QA/QC) procedures are
critical to ensure the integrity, accuracy, and reliability of the tidal
hindcast datasets. The project implements a comprehensive set of
verification processes to verify both individual files and the complete
dataset for each location.

## QA/QC Overview

| Category | Verification Process | Checked By | Is Compliant |
|----|----|----|----|
| Model Specification | Verification of model version, conventions, and required variables | NREL | Yes |
| Temporal Integrity | Validation of time steps, chronological ordering, and expected frequency | NREL | Yes |
| Spatial Consistency | Verification of coordinate systems and spatial grid consistency | NREL | Yes |
| Metadata Consistency | Validation of global attributes across files | NREL | Yes |
| Dataset Structure | Verification of dimension, variable, and coordinate consistency | NREL | Yes |
| Physical Validity | Range checks for physically meaningful values | PNNL |  |
| Completeness | Verification of temporal coverage and required variables | NREL | Yes |

## Automated Verification System

As the first step of data processing `src/verify.py` computes the
following checks for each location individually:

### 1. Model Specification Verification

- **Purpose**: Ensure files conform to expected FVCOM model
  specifications
- **Checks Performed**:
  - Validate model version matches expected value (FVCOM 4.3.1)
  - Verify conventions attribute (CF-1.0)
  - Confirm presence of all required variables
  - Validate variable data types and precision
  - Verify expected dimensions for each variable
  - Confirm required coordinates are present
  - Validate required variable attributes

### 2. Temporal Integrity Verification

- **Purpose**: Ensure consistent and complete time series data
- **Checks Performed**:
  - Verify time values always increase (monotonicity)
  - Validate time step frequency matches expected delta_t for location
  - Ensure no duplicate timestamps (with configurable handling strategy)
  - Check dataset spans expected temporal range from start_date_utc to
    end_date_utc
  - Verify consistent time steps throughout the dataset

### 3. Coordinate System Verification

- **Purpose**: Ensure consistent spatial representation
- **Checks Performed**:
  - Standardize coordinates according to location specifications
    (lat/lon or UTM)
  - Verify coordinate arrays match across all files for a location
  - Validate spatial coordinate integrity and consistency

#### Coordinate Transformation Specifications

#### Overview

The tidal hindcast dataset includes locations that use different
coordinate systems. A standardized transformation process ensures all
data is consistently represented in geographic coordinates
(latitude/longitude) for analysis and visualization.

## Coordinate Systems by Location

| Location         | Original Coordinate System | Transformation Required |
|------------------|----------------------------|-------------------------|
| Aleutian Islands | Geographic (lat/lon)       | No                      |
| Cook Inlet       | Geographic (lat/lon)       | No                      |
| Western Passage  | UTM Zone 19 (NAD83)        | Yes                     |
| Piscataqua River | UTM Zone 19 (NAD83)        | Yes                     |
| Puget Sound      | UTM Zone 10 (NAD83)        | Yes                     |

## Transformation Process

### Input Coordinate Reference Systems

- **Geographic Coordinates**: WGS84 datum (World Geodetic System 1984)
- **UTM Coordinates**: NAD83 datum (North American Datum 1983) with
  zone-specific projections
  - Western Passage: UTM Zone 19N
  - Piscataqua River: UTM Zone 19N
  - Puget Sound: UTM Zone 10N

### Output Coordinate Reference System

All data is transformed to a standardized geographic coordinate
system: - **Projection**: Latitude/Longitude - **Datum**: WGS84 -
**Ellipsoid**: WGS84 - **Angular Units**: Decimal degrees

### Transformation Implementation

The coordinate transformation process includes the following steps:

1.  **System Detection**:
    - Identify coordinate system from dataset attributes
      (`CoordinateSystem`)
    - Detect if transformation is required based on location
      configuration
2.  **Transformer Creation**:
    - For geographic coordinates (Aleutian Islands, Cook Inlet): No
      transformation required
    - For UTM coordinates (Western Passage, Piscataqua River, Puget
      Sound):
      - Create transformer from source UTM projection to WGS84
      - Source CRS defined by EPSG code based on UTM zone
3.  **Coordinate Processing**:
    - Transform node coordinates (original grid vertices)
    - Transform cell center coordinates (centroids of triangular
      elements)
    - Normalize longitudes to range \[-180, 180\] to handle date line
      crossing
4.  **Validation Procedures**:
    - Verify latitude values are within valid range \[-90, 90\]
    - Verify longitude values are within valid range \[-180, 180\]
    - Verify that cell centers lie within their triangular faces
    - Handle special cases for triangles crossing the international date
      line

## Projection String Limitations

While the original data files contain projection string information for
some locations, these strings could not be used directly due to several
issues:

1.  **Inconsistent Format**: The projection strings in the data files
    use non-standard formats and degree-minute notations (e.g., `-70d10`
    instead of decimal degrees `-70.16667`).

2.  **Incomplete Parameters**: Some projection strings lack important
    parameters required for accurate transformation.

3.  **UTM Zone Information**: For locations like Puget Sound, the
    projection string is listed as “none” despite using a UTM coordinate
    system.

4.  **Verification Challenges**: The original projection strings
    produced inconsistent results when plotted on maps.

Due to these limitations, the transformation process relies on
explicitly defined UTM zone parameters for each location rather than
parsing the projection strings from the data files. This approach
ensures consistent and accurate coordinate transformations across all
datasets.

## Coordinate Components

The standardized coordinate information includes:

| Component      | Description                                            |
|----------------|--------------------------------------------------------|
| lat_centers    | Latitude values at cell/element centers                |
| lon_centers    | Longitude values at cell/element centers               |
| lat_nodes      | Latitude values at mesh nodes/vertices                 |
| lon_nodes      | Longitude values at mesh nodes/vertices                |
| lat_face_nodes | Latitude values of nodes forming each triangular face  |
| lon_face_nodes | Longitude values of nodes forming each triangular face |

## Original Data Quality Report

| Label | Specification |
|----|----|
| Completeness | Full spatial and temporal coverage with few exceptions |
|  | Piscataqua River Missing Sigma Layer |
| Known Data Gaps | Puget Sound missing one day (2015-12-31) |
| Files with Issues | AK Aleutian Islands: MD_AIS_west_hrBathy_0370.nc (excluded) |
|  | AK Cook Inlet: cki_0366.nc (excluded) |
|  | NH Piscataqua River: PIR_0368.nc (excluded) |
| Quality Assessment Methods | FVCOM model validation against observational data |
| Flagging System | No flags in original data |

### 4. Global Attribute Equality Verification

- **Purpose**: Ensure consistent metadata across files
- **Checks Performed**:
  - Compare global attributes across files for consistency
  - Identify and report attribute mismatches
  - Allow configured exclusions for attributes that may vary (e.g.,
    history)

### 5. Dataset Structure Equality Verification

- **Purpose**: Ensure consistent structure across dataset files
- **Checks Performed**:
  - Verify consistent variable names across files
  - Validate dimension definitions match
  - Confirm coordinate definitions are consistent
  - Check attribute consistency for all components

## Quality Control Actions

The verification system implements several actions upon finding quality
issues:

| Action | Implementation | Details |
|----|----|----|
| Data Filtering | Exclusion of problematic files listed in configuration | Skip files manually specified in `config.py` |
| Verification Tracking | Generation of timestamped tracking files for each verification stage | Track timestamps for integrity and downstream processing |
| Temporal Coverage Validation | Verification of complete temporal coverage from start to end date |  |
| Time Series Repair | Identification of missing or duplicated timestamps | Use `"time_specification": "drop_duplicate_timestamps_keep_strategy": "first"` to handle duplicate timestamps |
| Spatial Integrity Validation | Confirmation of correct coordinate system implementation | Verify that face center points are within node coordinates. |

# Derived Variables, Value Added Products (VAP)

| Variable Name | Description | Formula | Units |
|----|----|----|----|
| vap_sea_water_speed | Sea Water Speed | $\sqrt{u^2 + v^2}$ | m s-1 |
| vap_sea_water_to_direction | Sea Water Velocity To Direction | $\text{mod}(90 - \text{atan2}(v, u) \cdot \frac{180}{\pi}, 360)$ | degree |
| vap_sea_water_power_density | Sea Water Power Density | $\frac{1}{2} \rho \cdot \text{speed}^3$ where $\rho = 1025$ kg/m³ | W m-2 |
| vap_zeta_center | Sea Surface Height at Cell Centers | Average of zeta values from three nodes of each face | m |
| vap_depth | Sigma Layer Depth Below Sea Surface | $-(h + \zeta) \times \sigma$ | m |
| vap_sea_floor_depth | Sea Floor Depth Below Sea Surface | $h + \zeta$ | m |
| vap_water_column_mean | Water Column Mean of a Variable | Mean across sigma layers | variable |
| vap_water_column_max | Water Column Maximum of a Variable | Maximum value across sigma layers | variable |
| vap_water_column_p95 | Water Column 95th Percentile of a Variable | Percentile value across sigma layers | variable |

# Data Quality Assurance and Quality Control

Data quality assurance and quality control (QA/QC) procedures are
critical to ensure the integrity, accuracy, and reliability of the tidal
hindcast datasets. The project implements a comprehensive set of
verification processes to validate both individual files and the
complete dataset for each location.

## QA/QC Overview

| Category | Verification Process | Checked By | Is Compliant |
|----|----|----|----|
| Model Specification | Verification of model version, conventions, and required variables | NREL | Yes |
| Temporal Integrity | Validation of time steps, chronological ordering, and expected frequency | NREL | Yes |
| Spatial Consistency | Verification of coordinate systems and spatial grid consistency | NREL | Yes |
| Metadata Consistency | Validation of global attributes across files | NREL | Yes |
| Dataset Structure | Verification of dimension, variable, and coordinate consistency | NREL | Yes |
| Physical Validity | Range checks for physically meaningful values | PNNL |  |
| Completeness | Verification of temporal coverage and required variables | NREL | Yes |

## Automated Verification System

As the first step of data processing `src/verify.py` computes the
following checks for each location individually:

### 1. Model Specification Verification

- **Purpose**: Ensure files conform to expected FVCOM model
  specifications
- **Checks Performed**:
  - Validate model version matches expected value (FVCOM 4.3.1)
  - Verify conventions attribute (CF-1.0)
  - Confirm presence of all required variables
  - Validate variable data types and precision
  - Verify expected dimensions for each variable
  - Confirm required coordinates are present
  - Validate required variable attributes

### 2. Temporal Integrity Verification

- **Purpose**: Ensure consistent and complete time series data
- **Checks Performed**:
  - Verify time values always increase (monotonicity)
  - Validate time step frequency matches expected delta_t for location
  - Ensure no duplicate timestamps (with configurable handling strategy)
  - Check dataset spans expected temporal range from start_date_utc to
    end_date_utc
  - Verify consistent time steps throughout the dataset

### 3. Coordinate System Verification

- **Purpose**: Ensure consistent spatial representation
- **Checks Performed**:
  - Standardize coordinates according to location specifications
    (lat/lon or UTM)
  - Verify coordinate arrays match across all files for a location
  - Validate spatial coordinate integrity and consistency

#### Coordinate Transformation Specifications

#### Overview

The tidal hindcast dataset includes locations that use different
coordinate systems. A standardized transformation process ensures all
data is consistently represented in geographic coordinates
(latitude/longitude) for analysis and visualization.

## Coordinate Systems by Location

| Location         | Original Coordinate System | Transformation Required |
|------------------|----------------------------|-------------------------|
| Aleutian Islands | Geographic (lat/lon)       | No                      |
| Cook Inlet       | Geographic (lat/lon)       | No                      |
| Western Passage  | UTM Zone 19 (NAD83)        | Yes                     |
| Piscataqua River | UTM Zone 19 (NAD83)        | Yes                     |
| Puget Sound      | UTM Zone 10 (NAD83)        | Yes                     |

## Transformation Process

### Input Coordinate Reference Systems

- **Geographic Coordinates**: WGS84 datum (World Geodetic System 1984)
- **UTM Coordinates**: NAD83 datum (North American Datum 1983) with
  zone-specific projections
  - Western Passage: UTM Zone 19N
  - Piscataqua River: UTM Zone 19N
  - Puget Sound: UTM Zone 10N

### Output Coordinate Reference System

All data is transformed to a standardized geographic coordinate
system: - **Projection**: Latitude/Longitude - **Datum**: WGS84 -
**Ellipsoid**: WGS84 - **Angular Units**: Decimal degrees

### Transformation Implementation

The coordinate transformation process includes the following steps:

1.  **System Detection**:
    - Identify coordinate system from dataset attributes
      (`CoordinateSystem`)
    - Detect if transformation is required based on location
      configuration
2.  **Transformer Creation**:
    - For geographic coordinates (Aleutian Islands, Cook Inlet): No
      transformation required
    - For UTM coordinates (Western Passage, Piscataqua River, Puget
      Sound):
      - Create transformer from source UTM projection to WGS84
      - Source CRS defined by EPSG code based on UTM zone
3.  **Coordinate Processing**:
    - Transform node coordinates (original grid vertices)
    - Transform cell center coordinates (centroids of triangular
      elements)
    - Normalize longitudes to range \[-180, 180\] to handle date line
      crossing
4.  **Validation Procedures**:
    - Verify latitude values are within valid range \[-90, 90\]
    - Verify longitude values are within valid range \[-180, 180\]
    - Verify that cell centers lie within their triangular faces
    - Handle special cases for triangles crossing the international date
      line

## Projection String Limitations

While the original data files contain projection string information for
some locations, these strings could not be used directly due to several
issues:

1.  **Inconsistent Format**: The projection strings in the data files
    use non-standard formats and degree-minute notations (e.g., `-70d10`
    instead of decimal degrees `-70.16667`).

2.  **Incomplete Parameters**: Some projection strings lack important
    parameters required for accurate transformation.

3.  **UTM Zone Information**: For locations like Puget Sound, the
    projection string is listed as “none” despite using a UTM coordinate
    system.

4.  **Verification Challenges**: The original projection strings
    produced inconsistent results when plotted on maps.

Due to these limitations, the transformation process relies on
explicitly defined UTM zone parameters for each location rather than
parsing the projection strings from the data files. This approach
ensures consistent and accurate coordinate transformations across all
datasets.

## Coordinate Components

The standardized coordinate information includes:

| Component      | Description                                            |
|----------------|--------------------------------------------------------|
| lat_centers    | Latitude values at cell/element centers                |
| lon_centers    | Longitude values at cell/element centers               |
| lat_nodes      | Latitude values at mesh nodes/vertices                 |
| lon_nodes      | Longitude values at mesh nodes/vertices                |
| lat_face_nodes | Latitude values of nodes forming each triangular face  |
| lon_face_nodes | Longitude values of nodes forming each triangular face |

## Original Data Quality Report

| Label | Specification |
|----|----|
| Completeness | Full spatial and temporal coverage with few exceptions |
|  | Piscataqua River Missing Sigma Layer |
| Known Data Gaps | Puget Sound missing one day (2015-12-31) |
| Files with Issues | AK Aleutian Islands: MD_AIS_west_hrBathy_0370.nc (excluded) |
|  | AK Cook Inlet: cki_0366.nc (excluded) |
|  | NH Piscataqua River: PIR_0368.nc (excluded) |
| Quality Assessment Methods | FVCOM model validation against observational data |
| Flagging System | No flags in original data |

### 4. Global Attribute Equality Verification

- **Purpose**: Ensure consistent metadata across files
- **Checks Performed**:
  - Compare global attributes across files for consistency
  - Identify and report attribute mismatches
  - Allow configured exclusions for attributes that may vary (e.g.,
    history)

### 5. Dataset Structure Equality Verification

- **Purpose**: Ensure consistent structure across dataset files
- **Checks Performed**:
  - Verify consistent variable names across files
  - Validate dimension definitions match
  - Confirm coordinate definitions are consistent
  - Check attribute consistency for all components

## Quality Control Actions

The verification system implements several actions upon finding quality
issues:

| Action | Implementation | Details |
|----|----|----|
| Data Filtering | Exclusion of problematic files listed in configuration | Skip files manually specified in `config.py` |
| Verification Tracking | Generation of timestamped tracking files for each verification stage | Track timestamps for integrity and downstream processing |
| Temporal Coverage Validation | Verification of complete temporal coverage from start to end date |  |
| Time Series Repair | Identification of missing or duplicated timestamps | Use `"time_specification": "drop_duplicate_timestamps_keep_strategy": "first"` to handle duplicate timestamps |
| Spatial Integrity Validation | Confirmation of correct coordinate system implementation | Verify that face center points are within node coordinates. |

## Partition Specifications

| Location         | Partition Frequency | Approximate Size per Partition |
|------------------|---------------------|--------------------------------|
| Aleutian Islands | 5D (5 days)         | 30 GB per partition            |
| Cook Inlet       | M (Monthly)         | ~35GB per partition            |
| Western Passage  | M (Monthly)         | ~50GB per partition            |
| Piscataqua River | M (Monthly)         | ~67GB per partition            |
| Puget Sound      | 5D (5 days)         | 30 GB per partition            |

# Data Levels and Processing Pipeline

The processing pipeline for the tidal hindcast data follows a structured
approach with clearly defined data levels. Each level represents a
specific stage of processing, with increasing value added as data
progresses through the pipeline.

## Data Level Definitions

| Data Level | Code | Description |
|----|----|----|
| Raw | `00_raw` | Original FVCOM model outputs with no modifications |
| Level a1 | `a1_std` | Standardized data with consistent variable naming, attributes, and quality control |
| Level a2 | `a2_std_partition` | Standardized data partitioned by time intervals for efficient processing |
| Level b1 | `b1_vap` | Value-added products including derived variables (velocity, power density, etc.) |
| Level b2 | `b2_summary_vap` | Statistical summaries across the vertical water column |
| Level b3 | `b3_vap_partition` | Partitioned VAP data for further processing |
| Level b5 | `b5_vap_summary_parquet` | Summary statistics in Parquet format optimized for data analytics |
| Level b6 | `b6_vap_atlas_summary_parquet` | NREL Marine Energy Atlas-ready summary data in Parquet format for visualization |

## Detailed Data Level Specifications

### Original/Raw Data (`00_raw`)

- **Description**: Original model output files from FVCOM in NetCDF
  format
- **Processing**: None, direct outputs from FVCOM model version 4.3.1
- **Variables**: Original variable names and attributes from FVCOM
  (time, lat, lon, u, v, zeta, etc.)
- **Dimensions**: Original mesh dimensions (node, nele, siglev, siglay)
- **File Format**: NetCDF4
- **Temporal Coverage**: Full modeled time period, varies by location (1
  year)
- **File Naming**: Original FVCOM model output naming
- **Storage**: Original data archived on AWS
- **Access**: Project use only, not public

### Level a1: Standardized Data (`a1_std`)

- **Description**: Standardized data with consistent variable naming and
  quality control
- **Processing**:
  - Standardized variable names and attributes according to CF
    conventions
  - Coordinate transformations (if applicable, e.g., UTM to lat/lon)
  - Time coordinate conversion to standard datetime format
  - Basic quality control (range checking, duplicate removal)
  - Variable metadata standardization
- **Variables**: Same as raw data but with standardized names and
  attributes
- **File Format**: NetCDF4, `*.nc`
- **Temporal Coverage**: Full modeled time period, varies by location (1
  year)
- **File Naming**: Following Marine Energy Data Pipeline convention
- **Storage**: HPC project storage
- **Access**: Project use only, not public

### Level a2: Standardized Partitioned Data (a2_std_partition)

- **Description**: Standardized data partitioned into specified time
  chunks for efficient processing
- **Processing**:
  - Time-based partitioning of a1 data into manageable chunks
  - Frequency varies by location (7D, 5D, or monthly)
- **Variables**: Same as a1 level
- **File Format**: NetCDF4, `*.nc`
- **Temporal Coverage**: Location-specific partitions of full time
  period
- **File Naming**: Marine Energy Data Pipeline convention with partition
  identifiers
- **Storage**: HPC project storage, Archived to AWS Long Term Storage
- **Access**: Project use only, not public

### Level b1: Value-Added Products (b1_vap)

- **Description**: Engineering focused dataset with derived variables
  for tidal energy assessment
- **Processing**:
  - Calculation of sea water speed (vector magnitude from u, v)
  - Calculation of flow direction (to/from directions)
  - Calculation of power density (½ρv³)
  - Calculation of element volumes
  - Calculation of volume flux
  - Calculation of depth at sigma levels
  - Interpolation of surface elevation to element centers
- **New Variables**:
  - `vap_sea_water_speed`
  - `vap_sea_water_to_direction`
  - `vap_sea_water_power_density`
  - `vap_element_volume`
  - `vap_zeta_center`
  - `vap_sigma_depth`
  - `vap_sea_floor_depth`
  - `vap_water_column_mean_u`
  - `vap_water_column_mean_v`
  - `vap_water_column_mean_to_direction`
  - `vap_water_column_mean_sea_water_speed`
  - `vap_water_column_max_sea_water_speed`
  - `vap_water_column_p95_sea_water_speed`
  - `vap_water_column_mean_sea_water_power_density`
  - `vap_water_column_max_sea_water_power_density`
  - `vap_water_column_p95_sea_water_power_density`
- **File Format**: NetCDF4
- **Temporal Coverage**: Same as a2 partitions
- **File Naming**: Marine Energy Data Pipeline convention
- **Storage**: HPC project storage and OpenEI AWS
- **Access**: Public access on OpenEI AWS

### Level b2: Summary Statistics (`b2_summary_vap`)

- **Description**: Statistical summaries of the VAP data across the
  water column
- **Processing**:
  - Calculation of water column statistics for key variables:
    - Mean (average across all sigma layers)
    - Median
    - Maximum values
    - 95th percentile values
- **New Variables**:
  - water_column_mean_sea_water_speed
  - water_column_median_sea_water_speed
  - water_column_max_sea_water_speed
  - water_column_95th_percentile_sea_water_speed
  - water_column_mean_sea_water_power_density
  - water_column_median_sea_water_power_density
  - water_column_max_sea_water_power_density
  - water_column_95th_percentile_sea_water_power_density
  - water_column_mean_eastward_velocity
  - water_column_mean_northward_velocity
  - water_column_mean_to_direction
  - water_column_mean_from_direction
- **File Format**: NetCDF4
- **Temporal Coverage**: Full modeled time period, varies by location (1
  year)
- **File Naming**: Marine Energy Data Pipeline convention
- **Storage**: HPC project storage and OpenEI AWS
- **Access**: Public access on OpenEI AWS

### Level b3: VAP Partitioned Data (`b3_vap_partition`)

- **Description**: Partitioned value-added products in individual files
  by location
- **Processing**: Time-based partitioning of b1 data
- **Variables**: Same as b1 level
- **File Format**: NetCDF4
- **Temporal Coverage**: Location-specific partitions of full time
  period
- **File Naming**: Marine Energy Data Pipeline convention with partition
  identifiers
- **Storage**: HPC project storage
- **Access**: Project use only, not public

### Level b4: Summary Parquet Files (b4_vap_summary_parquet)

- **Description**: Summary statistics in Parquet format with depth
  output as columnar data `<u_sigma_layer_n>`, etc
- **Processing**:
  - Conversion of b2 summary data to Parquet format
  - Optimization of table schema for analytics queries
- **Variables**: Same as b2 level but in tabular format
- **File Format**: Apache Parquet
- **Temporal Coverage**: Full modeled time period, varies by location (1
  year)
- **File Naming**: Marine Energy Data Pipeline convention with parquet
  extension
- **Storage**: HPC project storage and OpenEI AWS
- **Access**: Public access on OpenEI AWS

### Level b5: Atlas Summary Parquet Files (b5_vap_atlas_summary_parquet)

- **Description**: Specially formatted summary data for the Marine
  Energy Atlas visualization platform
- **Processing**:
  - Processing of summary statistics for Atlas-specific requirements
  - Optimization for web-based visualization
  - Aggregation of key metrics for interactive maps and charts
- **Variables**: Selected summary variables optimized for Atlas
  visualization
- **File Format**: Apache Parquet
- **Temporal Coverage**: Full modeled time period, varies by location (1
  year)
- **File Naming**: Marine Energy Data Pipeline convention with atlas
  identifier
- **Storage**: HPC project storage and OpenEI AWS
- **Access**: Public access on OpenEI AWS

## Processing Workflow

The data processing follows a sequential workflow:

1.  **Raw Data**: Original FVCOM model outputs
2.  **Standardization**: Conversion to a1_std with consistent naming and
    attributes
3.  **Partitioning**: Division into manageable time chunks
    (a2_std_partition)
4.  **Derivation**: Calculation of value-added variables (b1_vap)
5.  **Statistical Analysis**: Computation of summary statistics
    (b2_summary_vap)
6.  **Format Conversion**: Conversion to Parquet for analytics (b5) and
    Atlas visualization (b6)

Each step in the workflow adds value to the dataset, either through
standardization, derivation of new variables, statistical analysis, or
optimization for specific use cases.

## Directory Structure

The project follows a structured directory organization with clearly
defined paths for each data processing level. All outputs are organized
by location.

| Label | Path | Description |
|----|----|----|
| Tracking | `/projects/hindcastra/Tidal/datasets/<location>/z99_tracking` | Logs and processing metadata |
| Standardized (a1) | `/projects/hindcastra/Tidal/datasets/<location>/a1_std` | Standardized data with quality control |
| Standardized Partition (a2) | `/projects/hindcastra/Tidal/datasets/<location>/a2_std_partition` | Standardized data partitioned for efficient processing |
| VAP (b1) | `/projects/hindcastra/Tidal/datasets/<location>/b1_vap` | Value-added products including derived variables |
| Summary VAP (b2) | `/projects/hindcastra/Tidal/datasets/<location>/b2_summary_vap` | Summary statistics of value-added products |
| VAP Partition (b3) | `/projects/hindcastra/Tidal/datasets/<location>/b3_vap_partition` | Value-added products partitioned by time chunks |
| VAP Summary Parquet (b5) | `/projects/hindcastra/Tidal/datasets/<location>/b5_vap_summary_parquet` | Summary data in Parquet format for analytics |
| VAP Atlas Summary (b6) | `/projects/hindcastra/Tidal/datasets/<location>/b6_vap_atlas_summary_parquet` | Atlas-ready summary data in Parquet format |

## File Naming Convention

All output files follow the Marine Energy Data Pipeline naming
convention, which ensures consistency and traceability across all data
products.

### Data Level File Format

    location_id.dataset_name[-qualifier][-temporal].data_level.date.time.ext

Where: - `location_id`: Identifier for the location (e.g.,
AK_aleutian_islands) - `dataset_name`: Type of data (e.g.,
tidal_hindcast_fvcom) - `qualifier`: Optional string to distinguish
datasets from the same instrument - `temporal`: Temporal resolution
(e.g., 1h, 30m) - `data_level`: Two-character descriptor indicating
processing level (a1, b1, etc.) - `date`: Date in YYYYMMDD format -
`time`: Time in HHMMSS format - `ext`: File extension (nc for NetCDF,
parquet for Parquet files)

## Data Levels

| Data Level | Description | Format | Storage Location | Public Access |
|----|----|----|----|----|
| `00_raw` | Original NetCDF files from FVCOM model | NetCDF (nc) | HPC Kestrel / AWS Archive | No |
| `a1_std` | Standardized data with consistent naming and attributes | NetCDF (nc) | HPC Kestrel | No |
| `a2_std_partition` | Standardized data partitioned by time chunks for processing | NetCDF (nc) | HPC Kestrel | No |
| `b1_vap` | Value-added products with derived variables, full temporal resolution | NetCDF (nc) | HPC Kestrel / OpenEI AWS | Yes |
| `b2_summary_vap` | Summary statistics of value-added products | NetCDF (nc) | HPC Kestrel / OpenEI AWS | Yes |
| `b3_vap_partition` | Partitioned VAP data for efficient processing | NetCDF (nc) | HPC Kestrel / OpenEI AWS | Yes |
| `b4_vap_summary_parquet` | Summary data in Parquet format for analytics | Parquet | HPC Kestrel / OpenEI AWS | Yes |
| `b5_vap_atlas_summary_parquet` | Atlas-ready summary data for visualization | Parquet | HPC Kestrel / OpenEI AWS | Yes |

# Running Standardization Code

## Usage

``` bash
python runner.py <location>
```

``` bash
sbatch runner_cook_inlet.sbatch
```

# Included Metadata

<div id="tbl-included-metadata">

<div class="cell-output cell-output-display cell-output-markdown">

| Label | Key | Value |
|:---|:---|:---|
| Conventions | Conventions | CF-1.0, ACDD-1.3, ME Data Pipeline-1.0 |
| Acknowledgement | acknowledgement | This work was funded by the U.S. Department of Energy, Office of Energy Efficiency & Renewable Energy, Water Power Technologies Office. The authors gratefully acknowledge project support from Heather Spence and Jim McNally (U.S. Department of Energy Water Power Technologies Office) and Mary Serafin (National Renewable Energy Laboratory). Technical guidance was provided by Vincent Neary (Sandia National Laboratories), Levi Kilcher, Caroline Draxl, and Katie Peterson (National Renewable Energy Laboratory). |
| Creator Country | creator_country | USA |
| Creator Email | creator_email | zhaoqing.yang@pnnl.gov |
| Creator Institution | creator_institution | Pacific Northwest National Laboratory (PNNL) |
| Creator Institution Url | creator_institution_url | https://www.pnnl.gov/ |
| Creator Name | creator_name | Zhaoqing Yang |
| Creator Sector | creator_sector | gov_federal |
| Creator State | creator_state | Washington |
| Creator Type | creator_type | institution |
| Creator Url | creator_url | https://www.pnnl.gov/ |
| Contributor Name | contributor_name | Mithun Deb, Andrew Simms, Ethan Young |
| Contributor Role | contributor_role | author, processor, processor |
| Contributor Role Vocabulary | contributor_role_vocabulary | https://vocab.nerc.ac.uk/collection/G04/current/ |
| Contributor Url | contributor_url | https://www.pnnl.gov, www.nrel.gov |
| Featuretype | featureType | timeSeries |
| Infourl | infoURL | https://www.github.com/nrel/marine_energy_resource_characterization/tidal/fvcom/fy_25_tidal_hindcast |
| Keywords | keywords | OCEAN TIDES, TIDAL ENERGY, VELOCITY, SPEED, DIRECTION, POWER DENSITY |
| License | license | Freely Distributed |
| Naming Authority | naming_authority | gov.nrel.water_power |
| References | references | Spicer, P., Z. Yang, T. Wang, and M. Deb. (in prep.).Considering the relative importance of diurnal and semidiurnal tides in tidal power potential around the Aleutian Islands, AK, Renewable Energy. |
| Program | program | DOE Water Power Technologies Office Marine Energy Resource Characterization |
| Project | project | High Resolution Tidal Hindcast |
| Publisher Country | publisher_country | USA |
| Publisher Email | publisher_email | michael.lawson@nrel.gov |
| Publisher Institution | publisher_institution | Pacific Northwest National Laboratory (PNNL) |
| Publisher Name | publisher_name | Michael Lawson |
| Publisher State | publisher_state | Colorado |
| Publisher Type | publisher_type | institution |
| Publisher Url | publisher_url | https://www.nrel.gov |

</div>

Table 9: Included Metadata

</div>

# Visualization Specification

## Locations

Base Path: `/projects/hindcastra/Tidal/datasets/<location>/`

| Name | File Location | Format | Type |
|----|----|----|----|
| Aleutian Islands, Alaska | `/b5_vap_atlas_summary_parquet/AK_aleutian_islands.tidal_hindcast_fvcom-year_average.b5.20100603.000000.parquet` | Parquet | Yearly Average Depth Average |
| Cook Inlet, Alaska | `/b5_vap_atlas_summary_parquet/AK_cook_inlet.tidal_hindcast_fvcom-year_average.b5.20050101.000000.parquet` | Parquet | Yearly Average Depth Average |
| Western Passage, Maine | `/b5_vap_atlas_summary_parquet/ME_western_passage.tidal_hindcast_fvcom-year_average.b5.20170101.000000.parquet` | Parquet | Yearly Average Depth Average |
| Piscataqua River, New Hampshire | `/b5_vap_atlas_summary_parquet/NH_piscataqua_river.tidal_hindcast_fvcom-year_average.b5.20070101.000000.parquet` | Parquet | Yearly Average Depth Average |

## Output Variables

## Output Variable Color Bar Range

| Atlas Label | Units | Variable | Color Bar Min | Color Bar Max | Discrete Steps |
|----|----|----|----|----|----|
| Mean Sea Water Speed | meters per second, \[m s-1\] | `vap_water_column_mean_sea_water_speed` | 0 | 2 | 256 |
| Max Sea Water Speed | meters per second, \[m s-1\] | `vap_water_column_max_sea_water_speed` | 0 | 2 | 256 |
| Mean Sea Water Power Density | watts per meter squared, \[W m-2\] | `vap_water_column_mean_sea_water_power_density` | 0 | 8000 | 256 |
| Max Sea Water Power Density | watts per meter squared, \[W m-2\] | `vap_water_column_max_sea_water_speed` | 0 | 8000 | 256 |

# Acknowledgement

This work was funded by the U.S. Department of Energy, Office of Energy
Efficiency & Renewable Energy, Water Power Technologies Office. The
authors gratefully acknowledge project support from Heather Spence and
Jim McNally (U.S. Department of Energy Water Power Technologies Office)
and Mary Serafin (National Renewable Energy Laboratory). Technical
guidance was provided by Vincent Neary (Sandia National Laboratories),
Levi Kilcher, Caroline Draxl, and Katie Peterson (National Renewable
Energy Laboratory).

# Citation

# References

<!--
## Description
&#10;What is this data in two sentences
&#10;Original model output is stored in NREL managed long term storage for a period of X years
&#10;Standardized data transforms raw model output data into a common format and includes:
&#10;* Time validation and conversion to a standard format
* Coordinate validation and conversion to a standard format
* Addition of calculated engineering values specific to tidal energy development including:
    * Flow Velocity
        * Averaged and at each depth layer
    * Flow Direction
        * Averaged and at each depth layer
    * Power Density
        * Averaged and at each depth layer
    * Depth
* Organization of files my month
* Common metadata descriptions (`attrs`) in all files detailing data generation specifications,
  conventions, and other relevant information
* Unified variable is all datasets including units, name, and descriptions
&#10;The development of this dataset was funded by the U.S. Department of Energy,
Office of Energy Efficiency & Renewable Energy, Water Power Technologies Office
to improve our understanding of the U.S. tidal energy resource and to provide
critical information for tidal energy project development and tidal energy
converter design.
&#10;## Data Format
&#10;Standardized data is provided in netCDF4 format.
&#10;
## Available Data
&#10;The following U.S. regions will be added to this dataset under the given `domain` names:
&#10;| Dataset                 | Duration      | Frequency        | Start                    | End                      | Count         |
| ----------------------- | ------------- | ---------------- | ------------------------ | ------------------------ | ------------- |
| `AK_Aleutian_Islands`   | 365 Days      | Hourly           | 2010-06-03 00:00         | 2011-06-02 23:30         | 8760          |
| `AK_Cook_Inlet`         | 365 Days      | Hourly           | 2005-01-01 00:00         | 2005-12-31 23:30         | 8760          |
| `ME_Western_Passage`    | 365 Days      | Half Hourly      | 2017-01-01 00:00         | 2017-12-31 23:30         | 17520         |
| `NH_Piscataqua River`   | 365 Days      | Half Hourly      | 2007-01-01 00:00         | 2007-12-31 23:30         | 17520         |
| `WA_Puget_Sound`        | 365 Days      | Half Hourly      | 2015-01-01 00:00         | 2015-12-31 23:30         | 17520         |
&#10;
### Temporal Descriptions
&#10;| Dataset                 | Duration      | Frequency        | Start                    | End                      | Count         |
| ----------------------- | ------------- | ---------------- | ------------------------ | ------------------------ | ------------- |
| `AK_Aleutian_Islands`   | 365 Days      | Hourly           | 2010-06-03 00:00         | 2011-06-02 23:30         | 8760          |
| `AK_Cook_Inlet`         | 365 Days      | Hourly           | 2005-01-01 00:00         | 2005-12-31 23:30         | 8760          |
| `ME_Western_Passage`    | 365 Days      | Half Hourly      | 2017-01-01 00:00         | 2017-12-31 23:30         | 17520         |
| `NH_Piscataqua River`   | 365 Days      | Half Hourly      | 2007-01-01 00:00         | 2007-12-31 23:30         | 17520         |
| `WA_Puget_Sound`        | 365 Days      | Half Hourly      | 2015-01-01 00:00         | 2015-12-31 23:30         | 17520         |
&#10;### Location Descriptions
&#10;| Location               | Grid Type               | Avg Resolution    | Count         | Lat Min   | Lat Max   | Lon Min   | Lon Max   |
| ---------------------- | --------------------    | ----------------- | ------------- | --------- | --------- | --------- | --------- |
| `AK_Aleutian_Islands`  | Unstructured Triangular | `nan`             | `nan`         | `nan`     | `nan`     | `nan`     | `nan`     |
| `AK_Cook_Inlet`        | Unstructured Triangular | `nan`             | `nan`         | `nan`     | `nan`     | `nan`     | `nan`     |
| `ME_Western_Passages`  | Unstructured Triangular | `nan`             | `nan`         | `nan`     | `nan`     | `nan`     | `nan`     |
| `NH_Piscataqua River`  | Unstructured Triangular | `nan`             | `nan`         | `nan`     | `nan`     | `nan`     | `nan`     |
| `WA_Puget_Sound`       | Unstructured Triangular | `nan`             | `nan`         | `nan`     | `nan`     | `nan`     | `nan`     |
&#10;
### Coordinate Descriptions
&#10;Coordinates define the reference system for specifying spatial and temporal positions within a dataset. In this dataset, coordinates enable selection of data points by location and time, such as extracting a time series at a specific latitude and longitude that includes velocity across all depths.
&#10;
| Accessor | Name       | Units              | Description                          | Convention                                                                                              |
| -------  | ---------- | ------------------ | ------------------------------------ | ------------------------------------------------------------------------------------------------------ |
| `lat`    | Latitude   | Degrees North      | North-south position.                | [CF](https://cfconventions.org/cf-conventions/cf-conventions.html#latitude-coordinate)                 |
| `lon`    | Longitude  | Degrees East       | East-west position.                  | [CF](https://cfconventions.org/cf-conventions/cf-conventions.html#longitude-coordinate)                |
| `time`   | Time       | Seconds since 1970 | UTC time since Unix epoch.           | [CF](https://cfconventions.org/cf-conventions/cf-conventions.html#time-coordinate)                     |
| `depth`  | Depth      | Meter              | Vertical distance below the surface. | [CF](https://cfconventions.org/Data/cf-standard-names/current/build/cf-standard-name-table.html#depth) |
&#10;
### Variable Descriptions
&#10;The following variables are included in each dataset:
&#10;| Accessor            | Name                                    | Units | Dimensions            | Convention                                                                                                                    |
| ---                 | ---                                     | ---   | ---                   | ---                                                                                                                           |
| `u`                 | Eastward Water Velocity                 | m s-1 | time, lat, lon, depth | [`eastward_sea_water_velocity`](https://cfconventions.org/Data/cf-standard-names/current/build/cf-standard-name-table.html#eastward_sea_water_velocity)  |
| `v`                 | Northward Water Velocity                | m s-1 | time, lat, lon, depth | [`northward_sea_water_velocity`](https://cfconventions.org/Data/cf-standard-names/current/build/cf-standard-name-table.html#northward_sea_water_velocity) |
| `ww`                | Upward Water Velocity                   | m s-1 | time, lat, lon, depth | [`upward_sea_water_velocity`](https://cfconventions.org/Data/cf-standard-names/current/build/cf-standard-name-table.html#upward_sea_water_velocity)    |
| `vap_ua`            | Depth Averaged Eastward Water Velocity  | m s-1 | time, lat, lon        | [`eastward_sea_water_velocity`](https://cfconventions.org/Data/cf-standard-names/current/build/cf-standard-name-table.html#eastward_sea_water_velocity)  |
| `vap_va`            | Depth Averaged Northward Water Velocity | m s-1 | time, lat, lon        | [`northward_sea_water_velocity`](https://cfconventions.org/Data/cf-standard-names/current/build/cf-standard-name-table.html#northward_sea_water_velocity) |
| `vap_current_speed` | Calculated Current Speed                | m s-1 | time, lat, lon, depth | [`sea_water_speed`](https://cfconventions.org/Data/cf-standard-names/current/build/cf-standard-name-table.html#sea_water_speed) |
| `vap_power_density` | Depth Averaged Northward Water Velocity | m s-1 | time, lat, lon        | [CF](https://cfconventions.org/Data/cf-standard-names/current/build/cf-standard-name-table.html#northward_sea_water_velocity) |
&#10;
## Model Specifications
&#10;| Attribute                               | Description                                                                                         |
| ------------------------                | --------------------------------------------------------------------------------------------------- |
| Model                                   | [FVCOM 4.3.1](https://www.fvcom.org/)                                                                     |
| Dimensions                              | 3D                                                                                                  |
| Horizontal Resolution                   | Grid node and element points                                                                        |
| Vertical Resolution                     | 10 uniform sigma-stretched coordinates                                                              |
| Horizontal Coordinates                  | Latitude and longitude, UTM zone, State Plane                                                       |
| Vertical Datum                          | Mean Sea Level (MSL), NAVD88                                                                        |
| Temporal Resolution                     | Hourly, three hourly, daily                                                                         |
| Wetting & Drying Feature                | Activated                                                                                           |
| Boundary Forcing                        | 12 tidal constituents from OSU TPXO Tide Models                                                     |
| Wind                                    | ERA5 or CFSv2                                                                                       |
| IEC Standards                           | IEC TS 62600-201, 2015                                                                              |
| Wave-current interaction                | Not considered (small waves in study domain)                                                        |
| Atmospheric forcing                     | Not considered (wind and pressure effects negligible)                                               |
| Seawater density, salinity, temperature | Not considered (small estuarine flow in domain)                                                     |
&#10;
# Accessing Data
&#10;
## Python Examples
&#10;Example scripts to extract tidal data are.
&#10;The easiest way to access and extract data from the Resource eXtraction tool
[`rex`](https://github.com/nrel/rex)
&#10;To use `rex` with [`HSDS`](https://github.com/NREL/hsds-examples) you will need
to install `h5pyd`:
&#10;```
pip install h5pyd
```
&#10;Next you'll need to configure HSDS:
&#10;```
hsconfigure
```
&#10;and enter at the prompt:
&#10;```
hs_endpoint = https://developer.nrel.gov/api/hsds
hs_username =
hs_password =
hs_api_key = 3K3JQbjZmWctY0xmIfSYvYgtIcM3CN0cb1Y2w9bf
```
&#10;**IMPORTANT: The example API key here is for demonstation and is rate-limited per IP. To get your own API key, visit https://developer.nrel.gov/signup/**
&#10;You can also add the above contents to a configuration file at `~/.hscfg`
&#10;
```python
from rex import ResourceX
&#10;wave_file = '/nrel/US_wave/West_Coast/West_Coast_wave_2010.h5'
with ResourceX(wave_file, hsds=True) as f:
    meta = f.meta
    time_index = f.time_index
    swh = f['significant_wave_height']
```
&#10;`rex` also allows easy extraction of the nearest site to a desired (lat, lon)
location:
&#10;```python
from rex import ResourceX
&#10;wave_file = '/nrel/US_wave/West_Coast/West_Coast_wave_2010.h5'
lat_lon = (34.399408, -119.841181)
with ResourceX(wave_file, hsds=True) as f:
    lat_lon_swh = f.get_lat_lon_df('significant_wave_height', lat_lon)
```
&#10;or to extract all sites in a given region:
&#10;```python
from rex import ResourceX
&#10;wave_file = '/nrel/US_wave/West_Coast/West_Coast_wave_2010.h5'
jurisdication='California'
with ResourceX(wave_file, hsds=True) as f:
    ca_swh = f.get_region_df('significant_wave_height', jurisdiction,
                             region_col='jurisdiction')
```
&#10;If you would rather access the US Wave data directly using h5pyd:
&#10;```python
# Extract the average wave height
import h5pyd
import pandas as pd
&#10;# Open .h5 file
with h5pyd.File('/nrel/US_wave/West_Coast/West_Coast_wave_2010.h5', mode='r') as f:
    # Extract meta data and convert from records array to DataFrame
    meta = pd.DataFrame(f['meta'][...])
    # Significant Wave Height
    swh = f['significant_wave_height']
    # Extract scale factor
    scale_factor = swh.attrs['scale_factor']
    # Extract, average, and unscale wave height
    mean_swh = swh[...].mean(axis=0) / scale_factor
&#10;# Add mean wave height to meta data
meta['Average Wave Height'] = mean_swh
```
&#10;```python
# Extract time-series data for a single site
import h5pyd
import pandas as pd
&#10;# Open .h5 file
with h5pyd.File('/nrel/US_wave/West_Coast/West_Coast_wave_2010.h5', mode='r') as f:
    # Extract time_index and convert to datetime
    # NOTE: time_index is saved as byte-strings and must be decoded
    time_index = pd.to_datetime(f['time_index'][...].astype(str))
    # Initialize DataFrame to store time-series data
    time_series = pd.DataFrame(index=time_index)
    # Extract wave height, direction, and period
    for var in ['significant_wave_height', 'mean_wave_direction',
                'mean_absolute_period']:
        # Get dataset
        ds = f[var]
        # Extract scale factor
        scale_factor = ds.attrs['scale_factor']
        # Extract site 100 and add to DataFrame
        time_series[var] = ds[:, 100] / scale_factor
```
## References
&#10;Please cite the most relevant publication below when referencing this dataset:
&#10;1) [Wu, Wei-Cheng, et al. "Development and validation of a high-resolution regional wave hindcast model for US West Coast wave resource characterization." Renewable Energy 152 (2020): 736-753.](https://www.osti.gov/biblio/1599105)
2) [Yang, Z., G. García-Medina, W. Wu, and T. Wang, 2020. Characteristics and variability of the Nearshore Wave Resource on the U.S. West Coast. Energy.](https://doi.org/10.1016/j.energy.2020.117818)
3) [Yang, Zhaoqing, et al. High-Resolution Regional Wave Hindcast for the US West Coast. No. PNNL-28107. Pacific Northwest National Lab.(PNNL), Richland, WA (United States), 2018.](https://doi.org/10.2172/1573061)
4) [Ahn, S. V.S. Neary, Allahdadi, N. and R. He, Nearshore wave energy resource characterization along the East Coast of the United States, Renewable Energy, 2021, 172](https://doi.org/10.1016/j.renene.2021.03.037)
5) [Yang, Z. and V.S. Neary, High-resolution hindcasts for U.S. wave energy resource characterization. International Marine Energy Journal, 2020, 3, 65-71](https://doi.org/10.36688/imej.3.65-71)
6) [Allahdadi, M.N., He, R., and Neary, V.S.: Predicting ocean waves along the US East Coast during energetic winter storms: sensitivity to whitecapping parameterizations, Ocean Sci., 2019, 15, 691-715](https://doi.org/10.5194/os-15-691-2019)
7) [Allahdadi, M.N., Gunawan, J. Lai, R. He, V.S. Neary, Development and validation of a regional-scale high-resolution unstructured model for wave energy resource characterization along the US East Coast, Renewable Energy, 2019, 136, 500-511](https://doi.org/10.1016/j.renene.2019.01.020)
&#10;## Disclaimer and Attribution
&#10;The National Renewable Energy Laboratory (“NREL”) is operated for the U.S.
Department of Energy (“DOE”) by the Alliance for Sustainable Energy, LLC
("Alliance"). Pacific Northwest National Laboratory (PNNL) is managed and
operated by Battelle Memorial Institute ("Battelle") for DOE. As such the
following rules apply:
&#10;This data arose from worked performed under funding provided by the United
States Government. Access to or use of this data ("Data") denotes consent with
the fact that this data is provided "AS IS," “WHEREIS” AND SPECIFICALLY FREE
FROM ANY EXPRESS OR IMPLIED WARRANTY OF ANY KIND, INCLUDING BUT NOT LIMITED TO
ANY IMPLIED WARRANTIES SUCH AS MERCHANTABILITY AND/OR FITNESS FOR ANY
PARTICULAR PURPOSE. Furthermore, NEITHER THE UNITED STATES GOVERNMENT NOR ANY
OF ITS ASSOCITED ENTITES OR CONTRACTORS INCLUDING BUT NOT LIMITED TO THE
DOE/PNNL/NREL/BATTELLE/ALLIANCE ASSUME ANY LEGAL LIABILITY OR RESPONSIBILITY
FOR THE ACCURACY, COMPLETENESS, OR USEFULNESS OF THE DATA, OR REPRESENT THAT
ITS USE WOULD NOT INFRINGE PRIVATELY OWNED RIGHTS. NO ENDORSEMENT OF THE DATA
OR ANY REPRESENTATIONS MADE IN CONNECTION WITH THE DATA IS PROVIDED. IN NO
EVENT SHALL ANY PARTY BE LIABLE FOR ANY DAMAGES, INCLUDING BUT NOT LIMITED TO
SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES ARISING FROM THE PROVISION OF THIS
DATA; TO THE EXTENT PERMITTED BY LAW USER AGREES TO INDEMNIFY
DOE/PNNL/NREL/BATTELLE/ALLIANCE AND ITS SUBSIDIARIES, AFFILIATES, OFFICERS,
AGENTS, AND EMPLOYEES AGAINST ANY CLAIM OR DEMAND RELATED TO USER'S USE OF THE
DATA, INCLUDING ANY REASONABLE ATTORNEYS FEES INCURRED.
&#10;The user is granted the right, without any fee or cost, to use or copy the
Data, provided that this entire notice appears in all copies of the Data. In
the event that user engages in any scientific or technical publication
utilizing this data user agrees to credit DOE/PNNL/NREL/BATTELLE/ALLIANCE in
any such publication consistent with respective professional practice.
&#10;-->
