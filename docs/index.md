# Marine Energy Resource Characterization

Software to standardize wave and tidal energy model outputs for public access, supporting marine energy research, design, and deployment.

## Overview

This repository contains software that standardizes marine energy model data for public use. Developed collaboratively by [NLR](https://www.nlr.gov), [Sandia National Laboratories](https://www.sandia.gov), and [Pacific Northwest National Laboratory](https://www.pnnl.gov), our software converts complex ocean simulation outputs into standardized datasets that follow industry conventions ([IEC TC-114](https://www.iec.ch/dyn/www/f?p=103:7:::::FSP_ORG_ID:1316) and [CF standards](https://cfconventions.org)).

The project processes both wave energy ([SWAN](https://swanmodel.sourceforge.io)) and tidal energy ([FVCOM](http://fvcom.smast.umassd.edu/fvcom)) model outputs, making this valuable data accessible to researchers, developers, and policymakers.

## Multi-Laboratory Collaboration

The Marine Energy Resource Characterization project is a collaboration between:

- **[National Laboratory of the Rockies](https://www.nlr.gov)** (NLR) - Data processing and visualization
- **[Sandia National Laboratories](https://www.sandia.gov)** (SNL) - Data generation and validation
- **[Pacific Northwest National Laboratory](https://www.pnnl.gov)** (PNNL) - Tidal model development

This multi-laboratory initiative leverages high-performance computing to build computational fluid dynamics models of ocean conditions. The resulting simulations, combined with field measurements, aim to create a comprehensive understanding of the marine energy resource in the United States.

## Available Datasets

### [Wave Hindcast](wave-hindcast.md)

32-year high-resolution wave hindcast covering the U.S. Exclusive Economic Zone:

| Region     | Duration  | Resolution |
| ---------- | --------- | ---------- |
| West Coast | 1979-2010 | 3-hourly   |
| Atlantic   | 1979-2010 | 3-hourly   |
| Hawaii     | 1979-2010 | 3-hourly   |

### [Tidal Hindcast](tidal-hindcast.md)

High-resolution 3D tidal current data generated using FVCOM for five strategic U.S. coastal locations:

| Location                        | Duration | Resolution  | Grid Points |
| ------------------------------- | -------- | ----------- | ----------- |
| Aleutian Islands, Alaska        | 1 year   | Hourly      | 797,978     |
| Cook Inlet, Alaska              | 1 year   | Hourly      | 392,002     |
| Piscataqua River, New Hampshire | 1 year   | Half-hourly | 292,927     |
| Puget Sound, Washington         | 1 year   | Half-hourly | 1,734,765   |
| Western Passage, Maine          | 1 year   | Half-hourly | 231,208     |

## Applications

The standardized datasets support:

- Analysis of operational and extreme conditions at potential deployment sites
- Integration of modeled site-specific hydrodynamic forces into device design
- Public access to marine energy resource data through [OpenEI](https://openei.org/wiki/Marine_and_Hydrokinetic_Technology_Database)
- Visualization of summarized data on the [Marine Energy Atlas](https://maps.nlr.gov/marine-energy-atlas)

Quality control processes, standardized formatting, and documentation ensure the datasets are reliable and reproducible.

## Data Access

### AWS S3 Open Energy Data Initiative

Standardized datasets are publicly accessible through AWS S3:

- **Tidal Data**: [Marine Energy Data Lake](https://data.openei.org/s3_viewer?bucket=marine-energy-data)
- **Wave Data**: [WPTO PDS US Wave](https://registry.opendata.aws/wpto-pds-us-wave/)

### Marine Energy Atlas

Summary data is visualized on the [NLR Marine Energy Atlas](https://maps.nlr.gov/marine-energy-atlas), where select datasets can be downloaded using the point query tool.

### HSDS (Highly Scalable Data Service)

For programmatic access, data is available via HSDS at:

- Wave: `/nlr/US_wave/`
- Tidal: `/nlr/US_tidal/`

## Data Processing Workflow

```
Model Output → Verification → Standardization → Value-Added Products → Summary Statistics → Visualization
```

The processing pipeline ensures:

1. **Time Validation** - Consistent temporal formatting and completeness
2. **Coordinate Validation** - Standardized spatial reference systems
3. **Quality Control** - Range checking and anomaly detection
4. **Value-Added Products** - Derived engineering quantities (velocity, power density, direction)
5. **Documentation** - CF-compliant metadata and comprehensive documentation

## Standards Compliance

All datasets follow:

- **[CF Conventions](https://cfconventions.org)** - Climate and Forecast metadata standards
- **[IEC TS 62600-201](https://www.iec.ch/dyn/www/f?p=103:7:::::FSP_ORG_ID:1316)** - Tidal energy resource assessment
- **[ACDD](https://wiki.esipfed.org/Attribute_Convention_for_Data_Discovery_1-3)** - Attribute Convention for Data Discovery

## License

This software is licensed under the BSD 3-Clause License.

Copyright 2025 Alliance for Energy Innovation, LLC

This software was developed at least in part by Alliance for Energy Innovation, LLC ("Alliance") under Contract No. DE-AC36-08GO28308 with the U.S. Department of Energy and the U.S. Government retains for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in the software to reproduce, prepare derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so.

## Contact

- **GitHub Issues**: [Bug reports and feature requests](https://github.com/NatLabRockies/Marine_Energy_Resource_Characterization/issues)
- **Email**: [marineresource@nlr.gov](mailto:marineresource@nlr.gov)

## Acknowledgement

This work is funded by the U.S. Department of Energy, Water Power Technologies Office.
