# WPTO High Resolution Tidal Hindcast

High-resolution 3D tidal current hindcast data for U.S. coastal locations, generated using the Finite Volume Community Ocean Model (FVCOM) [@fvcom].

## Overview

This dataset contains high-resolution tidal hindcast data generated using FVCOM at five strategically selected U.S. coastal locations. The project represents a collaborative effort between [Pacific Northwest National Laboratory](https://www.pnnl.gov/marine-energy-resource-characterization) (PNNL) for data generation and the [National Laboratory of the Rockies](https://www.nlr.gov/water/resource-characterization) for data processing and visualization.

!!! info "Data Access"
    Complete standardized datasets are available from the [AWS S3 Open Energy Data Initiative Marine Energy Data Lake](https://data.openei.org/s3_viewer?bucket=marine-energy-data). Summary data is visualized on the [Marine Energy Atlas](https://maps.nlr.gov/marine-energy-atlas). See [Getting Started](../../getting-started/index.md) for HSDS/S3 setup and [Data Access](data-access.md) for dataset-specific code examples.

!!! warning "Data Limitations"
    This dataset is derived from numerical model simulations, not direct measurements. Results are based on a single hindcast year, which may not capture interannual variability in tidal energy resources. Model validation has been performed against observations at available measurement stations, but uncertainties exist, particularly in areas with complex bathymetry or limited observational data. See [Limitations](limitations.md) for details.

## Location Overview

| Location                        | Start Time [UTC]    | End Time [UTC]      | Sampling Frequency | Grid Points |
| ------------------------------- | ------------------- | ------------------- | ------------------ | ----------- |
| Aleutian Islands, Alaska        | 2010-06-03 00:00:00 | 2011-06-02 23:00:00 | Hourly             | 797,978     |
| Cook Inlet, Alaska              | 2005-01-01 00:00:00 | 2005-12-31 23:00:00 | Hourly             | 392,002     |
| Piscataqua River, New Hampshire | 2007-01-01 00:00:00 | 2007-12-31 23:30:00 | Half-Hourly        | 292,927     |
| Puget Sound, Washington         | 2015-01-01 00:00:00 | 2015-12-30 23:30:00 | Half-Hourly        | 1,734,765   |
| Western Passage, Maine          | 2017-01-01 00:00:00 | 2017-12-31 23:30:00 | Half-Hourly        | 231,208     |

## IEC 62600-201 Compliance

[@iec_62600_201]

| IEC Standards (IEC TS 62600-201, 2015)          | Compliance Status                            |
| ----------------------------------------------- | -------------------------------------------- |
| Stage 1 (feasibility) tidal resource assessment | Compliant with exceptions below              |
| Wave-current interaction                        | Not considered (waves small in study domain) |
| Atmospheric forcing                             | Not considered (effects negligible)          |
| Seawater density, salinity and temperature      | Not considered (density-induced flow small)  |

## Data Processing Levels

| Data Level | Description | Format | Public Access |
| ---------- | ----------- | ------ | ------------- |
| `00_raw` | Original FVCOM model outputs | NetCDF | No |
| `a1_std` | Standardized with consistent naming | NetCDF | No |
| `b1_vap` | Value-added products with derived variables | HDF5 | **Yes** |
| `b4_vap_summary_parquet` | Summary statistics for analytics | Parquet | **Yes** |
| `b5_vap_atlas_summary_parquet` | Marine Energy Atlas visualization data | Parquet | **Yes** |

## Citation

[!@mhkdr_submission]

## Acknowledgement

This work was funded by the U.S. Department of Energy's [Hydropower and Hydrokinetic Office (H2O)](https://www.energy.gov/cmei/water/hydropower-and-hydrokinetic-office). The authors gratefully acknowledge project support from Heather Spence and Jim McNally (U.S. Department of Energy Hydropower and Hydrokinetic Office (H2O)) and Mary Serafin (National Laboratory of the Rockies). Technical guidance was provided by Levi Kilcher, Caroline Draxl, and Katie Peterson (National Laboratory of the Rockies).


--8<-- "docs/tidal/high_resolution_hindcast/_cite-widget.md"
