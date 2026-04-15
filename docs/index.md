# Marine Energy Resource Characterization

Reconnaissance-level, high-resolution wave and tidal energy hindcast datasets for the United States — developed to [IEC 62600-101](https://webstore.iec.ch/en/publication/66062) [@iec_62600_101] and [IEC 62600-201](https://webstore.iec.ch/en/publication/22099) [@iec_62600_201] standards and publicly available through the [Marine and Hydrokinetic Data Repository (MHKDR)](https://mhkdr.openei.org) and the [Marine Energy Atlas](https://maps.nlr.gov/marine-energy-atlas).

!!! info "What is a hindcast?"
    A **hindcast** is a physics-based ocean simulation run over a historical time period using recorded atmospheric and oceanographic conditions as inputs. Rather than measuring the ocean directly, hindcasts use validated numerical models to reconstruct what conditions *would have been* at every location across a region — providing spatiotemporal coverage (data across both space and time simultaneously) that no observational network could practically achieve. The result is a consistent, gap-free record of wave heights, tidal currents, power density, and other variables across the U.S. coastline.

## About These Datasets

Developing marine energy requires detailed, reliable data: where is the resource, how strong is it, how does it vary across seasons and depth, and what are the conditions a device would actually experience? Generating that data requires large-scale physics-based ocean simulation — numerical models of tidal currents and ocean surface waves, validated against observations, and processed into standardized formats that engineers and policymakers can use with confidence.

These datasets are the result of that effort. Developed collaboratively by [NLR](https://www.nlr.gov), [Sandia National Laboratories](https://www.sandia.gov), and [Pacific Northwest National Laboratory](https://www.pnnl.gov) with funding from the U.S. Department of Energy's [Hydropower and Hydrokinetic Office (H2O)](https://www.energy.gov/cmei/water/hydropower-and-hydrokinetic-office), they represent a standardized, publicly accessible foundation for U.S. marine energy resource characterization.

## The Marine Energy Atlas and These Datasets

The [Marine Energy Atlas](https://maps.nlr.gov/marine-energy-atlas) is a visual exploration tool. It presents pre-computed summary layers — annual mean current speed, power density, tidal range, wave energy flux, and more — derived from these full hindcast datasets. The Atlas is the right starting point for identifying areas of interest and comparing resource potential across regions.

This documentation, and the underlying datasets, document and provide links to the data behind the Atlas layers:

- **Complete time series** at every model grid point — not just annual summaries
- **3D vertical structure** across depth layers (tidal) and spectral parameters (wave, virtual buoy)
- **IEC-compliant methodology** documentation for use in formal resource assessments
- **Direct data access** via [MHKDR](http://mhkdr.openei.org/632), [HSDS](getting-started/hsds-setup.md), and [AWS S3](getting-started/aws-s3.md)

## Who Is This For?

<div class="grid cards" markdown>

- **Policy Makers and Project Developers**

  ***

  If you are exploring where marine energy resources exist in the United States — asking questions like _"where are tidal currents strongest?"_ or _"what does the wave energy resource look like off the Atlantic coast?"_ — start with the [Marine Energy Atlas](https://maps.nlr.gov/marine-energy-atlas). The Atlas presents the summary data from these datasets in an interactive map you can explore without any technical background.

  This documentation explains what those Atlas layers mean and how the underlying data was produced.

  [Explore the Marine Energy Atlas](https://maps.nlr.gov/marine-energy-atlas){ .md-button }

  [About This Project](about/project.md)

- **Resource Analysts**

  ***

  If you need to compare resource potential across sites, understand summary statistics, or work with aggregated data for feasibility studies, the variable documentation is the right place to start. Each layer shown on the Atlas is documented here with definitions, equations, units, and known limitations.

  [Tidal Variables](tidal/high_resolution_hindcast/variables/index.md){ .md-button }

  [Wave Variables](wave/hindcast/variables.md)

  [Getting Started](getting-started/index.md)

- **Engineers and Researchers**

  ***

  If you need full time-series data for device design, structural loading, IEC Stage 1 resource assessments, or academic research, the complete hindcast datasets are available for download and programmatic access. Datasets follow CF-1.10, ACDD-1.3, IEC 62600-101 [@iec_62600_101], and IEC 62600-201 [@iec_62600_201] conventions and are fully documented including model configuration, validation, and known limitations.

  [Tidal Data Access](tidal/high_resolution_hindcast/data-access.md){ .md-button }

  [Wave Data Access](wave/hindcast/data-access.md)

  [Download Tidal Dataset on MHKDR](https://mhkdr.openei.org/submissions/632)

  [Download Wave Dataset on MHKDR](https://mhkdr.openei.org/submissions/326)

</div>

## Available Datasets

### [Tidal Hindcast](tidal/high_resolution_hindcast/index.md)

High-resolution 3D tidal current data for five U.S. coastal locations with significant tidal energy potential, generated using the Finite Volume Community Ocean Model (FVCOM). Each location covers one complete year at sub-500 m spatial resolution across 10 vertical depth layers — meeting IEC 62600-201 [@iec_62600_201] Stage 1 standards for tidal energy resource assessment.

| Location                        | Simulation Period | Temporal Resolution | Grid Points |
| ------------------------------- | ----------------- | ------------------- | ----------- |
| Aleutian Islands, Alaska        | 2010–2011         | Hourly              | 797,978     |
| Cook Inlet, Alaska              | 2005              | Hourly              | 392,002     |
| Piscataqua River, New Hampshire | 2007              | Half-hourly         | 292,927     |
| Puget Sound, Washington         | 2015              | Half-hourly         | 1,734,765   |
| Western Passage, Maine          | 2017              | Half-hourly         | 231,208     |

[Full tidal documentation](tidal/high_resolution_hindcast/index.md) · [Download on MHKDR](https://mhkdr.openei.org/submissions/632)

### [Wave Hindcast](wave/hindcast/index.md)

A 40-year (1979–2020) high-resolution wave hindcast covering the U.S. Exclusive Economic Zone (EEZ), produced using WaveWatch III and SWAN. Provides the long-term statistical record of wave conditions needed for IEC 62600-101 [@iec_62600_101] wave energy resource assessment and device design.

| Region                          | Simulation Period | Temporal Resolution |
| ------------------------------- | ----------------- | ------------------- |
| West Coast                      | 1979–2020         | 3-hourly            |
| Atlantic                        | 1979–2020         | 3-hourly            |
| Hawaii                          | 1979–2020         | 3-hourly            |
| Puerto Rico & Gulf of Mexico    | 1979–2020         | 3-hourly            |
| Guam & Northern Mariana Islands | 1979–2020         | 3-hourly            |

[Full wave documentation](wave/hindcast/index.md) · [Download on MHKDR](https://mhkdr.openei.org/submissions/326)

## Scientific Standards

All datasets are built to internationally recognized standards to ensure scientific rigor, consistency, and suitability for formal engineering assessments:

| Standard                                                                          | What It Ensures                                                                                        |
| --------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| [IEC TS 62600-101](https://webstore.iec.ch/en/publication/66062) [@iec_62600_101] | Wave energy resource assessment methodology, variable definitions, and reporting requirements          |
| [IEC TS 62600-201](https://webstore.iec.ch/en/publication/22099) [@iec_62600_201] | Tidal energy resource assessment methodology, variable definitions, and reporting requirements         |
| [CF Conventions](https://cfconventions.org)                                       | Consistent variable names, units, coordinate metadata, and grid conventions for ocean and climate data |
| [ACDD](https://wiki.esipfed.org/Attribute_Convention_for_Data_Discovery_1-3)      | Dataset-level metadata attributes for discoverability in scientific data catalogs                      |

Adherence to these standards means data can be directly applied in IEC-compliant resource assessments and integrated with other oceanographic and climate datasets without format conversion.

## Contact

- **GitHub Issues**: [Bug reports and feature requests](https://github.com/NatLabRockies/Marine_Energy_Resource_Characterization/issues)
- **Email**: [marineresource@nlr.gov](mailto:marineresource@nlr.gov)

<div class="bibliography"></div>
