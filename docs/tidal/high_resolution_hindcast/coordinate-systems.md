# Coordinate Systems

## Output Coordinate System

All data is distributed in **WGS84 (EPSG:4326)** geographic coordinates (latitude, longitude).

## Original Coordinate Systems

Some FVCOM model domains were originally configured in projected coordinate systems and were transformed during standardization:

| Location         | Original System      | Transformation |
| ---------------- | -------------------- | -------------- |
| Aleutian Islands | Geographic (lat/lon) | None           |
| Cook Inlet       | Geographic (lat/lon) | None           |
| Western Passage  | UTM Zone 19 (NAD83)  | → WGS84        |
| Piscataqua River | UTM Zone 19 (NAD83)  | → WGS84        |
| Puget Sound      | UTM Zone 10 (NAD83)  | → WGS84        |

## Transformation Details

UTM-to-WGS84 transformations were performed using [pyproj](https://pyproj4.github.io/pyproj/) with the following parameters:

- **Source CRS**: NAD83 UTM (Zone 10 or 19 depending on location)
- **Target CRS**: WGS84 (EPSG:4326)
- **Method**: Coordinate transformation via pyproj Transformer

## Vertical Datum

- **Model reference**: Mean Sea Level (MSL)
- **Vertical datum**: NAVD88 (North American Vertical Datum of 1988)
- **Convention**: Sea surface elevation ($\zeta$) is positive upward relative to NAVD88; bathymetry ($h$) is positive downward below NAVD88


--8<-- "docs/tidal/high_resolution_hindcast/_cite-widget.md"
