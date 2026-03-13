# Data Formats

## Processing Levels

Data progresses through multiple processing levels from raw model output to atlas-ready summaries:

| Data Level | Description | Format | Public Access |
| ---------- | ----------- | ------ | ------------- |
| `00_raw` | Original model outputs (FVCOM / WW3+SWAN) | NetCDF | No |
| `a1_std` | Standardized with consistent naming and metadata | NetCDF | No |
| `b1_vap` | Value-added products with derived variables | HDF5 | **Yes** |
| `b4_vap_summary_parquet` | Summary statistics for analytics | Parquet | **Yes** |
| `b5_vap_atlas_summary_parquet` | Marine Energy Atlas visualization data | Parquet | **Yes** |

## HDF5 Structure

Public datasets are distributed as HDF5 files (`.h5`). Each file contains:

- **`time_index`** — Timestamps as ISO 8601 strings
- **`meta`** — Structured array with latitude, longitude, and spatial metadata per grid point
- **Variable datasets** — 2D arrays with shape `(time, location)`, stored as scaled integers with a `scale_factor` attribute

Example file structure:

```
Cook_Inlet_2005.h5
├── time_index          (8760,)    # hourly timestamps for 1 year
├── meta                (392002,)  # grid point metadata
├── sea_water_speed     (8760, 392002)  # current speed at each timestep
├── sea_water_power_density  (8760, 392002)
└── ...
```

To retrieve physical values, divide by the scale factor:

```python
physical_value = dataset[...] / dataset.attrs['scale_factor']
```

## Parquet Format

Summary statistics are available as Parquet files for efficient analytical queries. Parquet files are partitioned by location and contain one row per grid point with columns for each summary variable.

## File Naming Conventions

### Tidal Files

```
{Location}_{Year}.h5
```

Examples: `Cook_Inlet_2005.h5`, `Puget_Sound_2015.h5`

### Wave Files

```
{Region}_wave_{Year}.h5
```

Examples: `West_Coast_wave_2010.h5`, `Atlantic_wave_1995.h5`

## Data Dimensions

### Tidal

| Dimension | Description |
| --------- | ----------- |
| time | Hourly or half-hourly timestamps over 1 year |
| element | Unstructured triangular grid cell centers |
| depth | 10 sigma layers (terrain-following vertical coordinate) |

### Wave

| Dimension | Description |
| --------- | ----------- |
| time | 3-hourly timestamps over 32 years |
| location | Unstructured grid points (200m–10km spacing) |
