# Model Description

## Multi-Scale Modeling Approach

The multi-scale, unstructured-grid modeling approach using WaveWatch III and SWAN enabled long-term (decades) high-resolution hindcasts in a large regional domain:

- **Outer Model**: WaveWatch III with global-regional nested grids
- **Inner Model**: Unstructured-grid SWAN with resolution as fine as 200 meters in shallow waters
- **Timestep**: 3-hour intervals spanning 32 years (1979-2010)

## Validation

The models were extensively validated against:

- Common wave parameters
- Six IEC resource parameters
- 2D spectra from high-quality spectral buoy data

## Data Format

The data is provided in high-density HDF5 files (.h5) separated by year:

- **Dimensions**: 2D time-series arrays (time x location)
- **Temporal Axis**: Defined by `time_index` dataset
- **Spatial Axis**: Defined by `coordinate` dataset
- **Attributes**: Units, SWAN names, IEC names included
