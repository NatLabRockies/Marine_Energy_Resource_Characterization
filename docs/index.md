# Marine Energy Resource Characterization

Standardized wave and tidal energy datasets for the United States, supporting marine energy research, design, and deployment.

## Overview

This project creates publicly accessible, standardized datasets describing the tidal and wave energy resources of the United States. Developed collaboratively by [NLR](https://www.nlr.gov), [Sandia National Laboratories](https://www.sandia.gov), and [Pacific Northwest National Laboratory](https://www.pnnl.gov), the project converts complex ocean simulation outputs into standardized datasets following [IEC TC-114](https://www.iec.ch/dyn/www/f?p=103:7:::::FSP_ORG_ID:1316) and [CF Conventions](https://cfconventions.org).

## Available Datasets

### [Wave Hindcast](wave/hindcast/index.md)

40-year high-resolution wave hindcast covering the U.S. Exclusive Economic Zone:

| Region                          | Duration  | Resolution |
| ------------------------------- | --------- | ---------- |
| West Coast                      | 1979-2020 | 3-hourly   |
| Atlantic                        | 1979-2020 | 3-hourly   |
| Hawaii                          | 1979-2020 | 3-hourly   |
| Puerto Rico and Gulf of Mexico  | 1979-2020 | 3-hourly   |
| Guam & Northern Mariana Islands | 1979-2020 | 3-hourly   |

### [Tidal Hindcast](tidal/high_resolution_hindcast/index.md)

1 year High-resolution 3D tidal current data generated using FVCOM for five strategic U.S. coastal locations:

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

## Standards Compliance

All datasets follow:

- **[CF Conventions](https://cfconventions.org)** — Climate and Forecast metadata standards
- **[IEC TS 62600-201](https://www.iec.ch/dyn/www/f?p=103:7:::::FSP_ORG_ID:1316)** — Tidal energy resource assessment
- **[ACDD](https://wiki.esipfed.org/Attribute_Convention_for_Data_Discovery_1-3)** — Attribute Convention for Data Discovery

## Quick Start

```python
from rex import ResourceX

# Access tidal data
with ResourceX('/nlr/US_tidal/Cook_Inlet/Cook_Inlet_2005.h5', hsds=True) as f:
    speed = f['sea_water_speed']
```

See [Getting Started](getting-started/index.md) for setup instructions.

## Contact

- **GitHub Issues**: [Bug reports and feature requests](https://github.com/NatLabRockies/Marine_Energy_Resource_Characterization/issues)
- **Email**: [marineresource@nlr.gov](mailto:marineresource@nlr.gov)
