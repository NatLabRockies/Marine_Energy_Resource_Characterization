# Changelog

## v1.0.0 — Initial Release (2025)

Initial public release of the WPTO High Resolution Tidal Hindcast dataset.

### Locations Released

- Aleutian Islands, Alaska (797,978 grid points, hourly, 2010–2011)
- Cook Inlet, Alaska (392,002 grid points, hourly, 2005)
- Piscataqua River, New Hampshire (292,927 grid points, half-hourly, 2007)
- Puget Sound, Washington (1,734,765 grid points, half-hourly, 2015)
- Western Passage, Maine (231,208 grid points, half-hourly, 2017)

### Variables Included

- Mean Current Speed (depth-averaged)
- 95th Percentile Current Speed (depth-maximum)
- Mean Power Density (depth-averaged)
- 95th Percentile Power Density (depth-maximum)
- Minimum Water Depth
- Maximum Water Depth
- Grid Resolution
- Tidal Range
- Distance to Shore
- Maximum Sea Surface Elevation at High Tide
- Minimum Sea Surface Elevation at Low Tide

### Data Formats

- HDF5 time-series files via HSDS and AWS S3
- Parquet summary statistics
- Marine Energy Atlas visualization layers

### Known Issues

- Puget Sound is missing December 31, 2015
- See [Quality Assurance](quality-assurance.md) for excluded files


--8<-- "docs/tidal/high_resolution_hindcast/_cite-widget.md"
