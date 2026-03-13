<!-- AUTO-GENERATED FILE — DO NOT EDIT DIRECTLY -->
<!-- Source of truth: src/variable_registry.py (VARIABLE_REGISTRY) -->
<!-- To update: edit the registry, then run `python generate_variable_docs.py` -->

# Variable Quick Reference

Summary of all variables in the High Resolution Tidal Hindcast dataset. Click a variable name to view its full documentation including description, equation, and properties.

| Variable | Internal Name | Units | Description |
| --- | --- | --- | --- |
| [Mean Current Speed](mean-current-speed.md) | `vap_water_column_mean_sea_water_speed` | m/s | Annual average of depth-averaged current speed |
| [95th Percentile Current Speed](95th-percentile-current-speed.md) | `vap_water_column_95th_percentile_sea_water_speed` | m/s | Estimated extreme current speed, outlier-tolerant and comparable across sites for reconnaissance-level assessment |
| [Mean Power Density](mean-power-density.md) | `vap_water_column_mean_sea_water_power_density` | W/m² | Annual average of depth-averaged kinetic energy flux |
| [95th Percentile Power Density](95th-percentile-power-density.md) | `vap_water_column_95th_percentile_sea_water_power_density` | W/m² | Estimated extreme power density, outlier-tolerant and robust to cubic-velocity sensitivity |
| [Minimum Water Depth](minimum-water-depth.md) | `vap_water_column_height_min` | m | Minimum water depth calculated over the 1-year hindcast period |
| [Maximum Water Depth](maximum-water-depth.md) | `vap_water_column_height_max` | m | Maximum water depth calculated over the 1-year hindcast period |
| [Grid Resolution](grid-resolution.md) | `vap_grid_resolution` | m | Average edge length of triangular model grid cells |
| [Tidal Range](tidal-range.md) | `vap_tidal_range` | m | Difference between maximum and minimum sea surface elevation over the hindcast year |
| [Distance to Shore](distance-to-shore.md) | `vap_distance_to_shore` | NM | Geodesic distance from grid cell center to nearest shoreline |
| [Maximum Sea Surface Elevation at High Tide](max-sea-surface-elevation-at-high-tide.md) | `vap_sea_surface_elevation_high_tide_max` | m (relative to model MSL) | Highest sea surface elevation calculated during high tide over the hindcast year |
| [Minimum Sea Surface Elevation at Low Tide](min-sea-surface-elevation-at-low-tide.md) | `vap_surface_elevation_low_tide_min` | m (relative to model MSL) | Lowest sea surface elevation calculated during low tide over the hindcast year |
| [Face ID](face_id.md) | `face_id` |  | Location specific unique integer identifier for each triangular grid element |
| [Center Latitude](center_latitude.md) | `lat_center` | degrees_north | Latitude of the triangular element centroid (WGS84) |
| [Center Longitude](center_longitude.md) | `lon_center` | degrees_east | Longitude of the triangular element centroid (WGS84) |
| [S3 URI for Full Year Time Series Data](full_year_s3_uri.md) | `full_year_data_s3_uri` |  | direct link (S3 URI) to download the one-year hindcast time series (parquet) for this location. Includes speed, direction, for 10 uniform sigma levels at half-hourly (lower 48) or hourly (Alaska) intervals. |
| [HTTPS URL for Full Year Time Series Data](full_year_https_url.md) | `full_year_data_https_url` |  | direct link (HTTPS)  to download the one-year hindcast time series (parquet) for this location. Includes speed, direction, for 10 uniform sigma levels at half-hourly (lower 48) or hourly (Alaska) intervals |

--8<-- "docs/tidal/high_resolution_hindcast/_cite-widget.md"
