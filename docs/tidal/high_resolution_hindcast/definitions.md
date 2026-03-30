# Definitions

Technical terms, mathematical symbols, and acronyms used throughout this documentation and in the dataset metadata.

## Technical Terms

**Hindcast**
:   A historical simulation of ocean conditions using a numerical model driven by observed atmospheric forcing and tidal boundary conditions. Unlike a forecast (which predicts future conditions) or direct measurements, a hindcast reconstructs past conditions by running the model over a historical time period. The results represent modeled estimates of what conditions were, not direct observations.

**Sigma Layer**
:   A terrain-following vertical coordinate system where the water column is divided into layers that conform to both the seafloor and the time-varying sea surface. See [Sigma Layers](sigma-layers.md) for details.

**Depth-Averaged**
:   A value computed by averaging a quantity across all vertical layers (sigma layers) at a given horizontal location and time. Produces a single representative value for the entire water column.

**Depth-Maximum**
:   The maximum value of a quantity across all vertical layers at a given horizontal location and time. Captures the peak value occurring anywhere in the water column.

**Free-Stream Velocity**
:   The undisturbed flow velocity that would exist in the absence of any energy extraction device. All values in this dataset represent free-stream conditions.

**Unstructured Grid**
:   A computational mesh composed of irregularly-shaped triangular elements. See [Unstructured Grid](unstructured-grid.md) for details.

## Symbols

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

## Acronyms

| Acronym        | Definition                                                                       |
| -------------- | -------------------------------------------------------------------------------- |
| FVCOM          | Finite Volume Community Ocean Model                                              |
| NAVD88         | North American Vertical Datum of 1988                                            |
| MSL            | Mean Sea Level                                                                   |
| IEC            | International Electrotechnical Commission                                        |
| IEC 62600-201  | International standard for tidal energy resource assessment and characterization |
| CF Conventions | Climate and Forecast Conventions                                                 |
| PNNL           | Pacific Northwest National Laboratory                                            |
| H2O            | Hydropower and Hydrokinetic Office                                               |
| VAP            | Value-Added Product                                                              |


--8<-- "docs/tidal/high_resolution_hindcast/_cite-widget.md"
