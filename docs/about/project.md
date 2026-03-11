# The Project

## Background

The Marine Energy Resource Characterization project is a multi-laboratory initiative that creates standardized, publicly accessible datasets describing the tidal and wave energy resources of the United States. By converting complex ocean simulation outputs into standardized formats, the project supports researchers, developers, and policymakers working to advance marine energy technology.

## Goals

- **Standardize** marine energy model outputs following [IEC TC-114 / 62600 Standards](https://www.iec.ch/dyn/www/f?p=103:7:::::FSP_ORG_ID:1316) and [Climate For.cast (CF) Conventions](https://cfconventions.org)
- **Publish** high-resolution hindcast datasets for public access through AWS S3 and HSDS
- **Visualize** summary data on the [Marine Energy Atlas](https://maps.nlr.gov/marine-energy-atlas)
- **Document** methodology, validation, and data quality for reproducibility

## Processing Pipeline

All datasets follow a standardized processing workflow:

```
Model Output → Verification → Standardization → Value-Added Products → Summary Statistics → Visualization
```

1. **Time Validation** — Consistent temporal formatting and completeness
2. **Coordinate Validation** — Standardized spatial reference systems
3. **Quality Control** — Range checking and anomaly detection
4. **Value-Added Products** — Derived engineering quantities (velocity, power density, direction)
5. **Documentation** — CF-compliant metadata and comprehensive documentation

## Multi-Laboratory Collaboration

This project leverages the complementary expertise of three U.S. Department of Energy national laboratories:

- **[Pacific Northwest National Laboratory](https://www.pnnl.gov)** (PNNL) — Tidal and wave model development, data generation, and validation
- **[National Laboratory of the Rockies](https://www.nlr.gov)** (NLR) — Data processing, standardization, visualization, and public access
- **[Sandia National Laboratories](https://www.sandia.gov)** (SNL) — Wave model development and data generation

Together, these laboratories leverage high-performance computing to build computational fluid dynamics models of ocean conditions. The resulting simulations, combined with field measurements, create a comprehensive understanding of the marine energy resource in the United States.

## Standards Compliance

All datasets follow:

- **[CF Conventions](https://cfconventions.org)** — Climate and Forecast metadata standards
- **[IEC TS 62600 Marine Energy Standards](https://www.iec.ch/dyn/www/f?p=103:7:::::FSP_ORG_ID:1316)** — Tidal energy resource assessment
- **[ACDD](https://wiki.esipfed.org/Attribute_Convention_for_Data_Discovery_1-3)** — Attribute Convention for Data Discovery

## License

This software is licensed under the BSD 3-Clause License.

Copyright 2025 Alliance for Energy Innovation, LLC

This software was developed at least in part by Alliance for Energy Innovation, LLC ("Alliance") under Contract No. DE-AC36-08GO28308 with the U.S. Department of Energy and the U.S. Government retains for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in the software to reproduce, prepare derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so.
