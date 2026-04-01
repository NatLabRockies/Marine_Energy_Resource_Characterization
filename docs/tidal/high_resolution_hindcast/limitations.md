# Limitations

This dataset is derived from numerical model simulations and has inherent limitations that users should understand when applying the data.

## Physics Not Included

The model does not include:

- **Wave-current interaction** — Waves are small in the study domains
- **Atmospheric forcing** — Wind and pressure effects on tidal currents are negligible
- **Density-driven estuarine flow** — Temperature and salinity effects are small
- **Storm surge** — Only astronomical tidal forcing from 12 constituents via OSU TPXO Tide Models is applied

## Temporal Limitations

Results represent a **single hindcast year** and do not capture interannual variability. Tidal patterns are highly predictable, but extreme values may vary between years due to meteorological effects.

## Spatial Limitations

- The model uses **wetting/drying algorithms** for intertidal zones; results in these areas should be interpreted with care
- Model accuracy may be **reduced near open ocean boundaries**
- Features smaller than approximately **2-3 times the local grid spacing** cannot be accurately resolved

## Known Data Gaps

| Location         | Issue                                     |
| ---------------- | ----------------------------------------- |
| Puget Sound      | Missing 2015-12-31                        |
| Aleutian Islands | File MD_AIS_west_hrBathy_0370.nc excluded |
| Cook Inlet       | File cki_0366.nc excluded                 |
| Piscataqua River | File PIR_0368.nc excluded                 |

See [Quality Assurance](quality-assurance.md) for details on data verification.

## Free-Stream Conditions

All velocity and power density values represent free-stream (undisturbed) conditions. They should **not** be used directly for turbine array yield estimation. See [Unstructured Grid](unstructured-grid.md#free-stream-velocity) for details on blockage and wake effects.


--8<-- "docs/tidal/high_resolution_hindcast/_cite-widget.md"
