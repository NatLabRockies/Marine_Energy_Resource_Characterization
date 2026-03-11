# AWS S3 Downloads

Standardized datasets are publicly accessible through the AWS Open Energy Data Initiative. No AWS account is required for browsing or downloading.

## Browse Online

- **Tidal Data**: [Marine Energy Data Lake](https://data.openei.org/s3_viewer?bucket=marine-energy-data)
- **Wave Data**: [WPTO PDS US Wave](https://registry.opendata.aws/wpto-pds-us-wave/)

## Bucket Structure

### Tidal Data

```
s3://marine-energy-data/
└── US_tidal/
    ├── Aleutian_Islands/
    ├── Cook_Inlet/
    ├── Piscataqua_River/
    ├── Puget_Sound/
    └── Western_Passage/
```

### Wave Data

```
s3://wpto-pds-US_wave/
└── v1.0.0/
    ├── West_Coast/
    ├── Atlantic/
    ├── Hawaii/
    └── virtual_buoy/
        ├── West_Coast/
        ├── Atlantic/
        └── Hawaii/
```

## Download via AWS CLI

Install the [AWS CLI](https://aws.amazon.com/cli/) and download files without authentication:

```bash
# List tidal datasets
aws s3 ls s3://marine-energy-data/US_tidal/ --no-sign-request

# Download a specific tidal file
aws s3 cp s3://marine-energy-data/US_tidal/Cook_Inlet/Cook_Inlet_2005.h5 . --no-sign-request

# Download an entire tidal location
aws s3 sync s3://marine-energy-data/US_tidal/Cook_Inlet/ ./Cook_Inlet/ --no-sign-request
```

```bash
# List wave datasets
aws s3 ls s3://wpto-pds-US_wave/v1.0.0/ --no-sign-request

# Download a specific wave file
aws s3 cp s3://wpto-pds-US_wave/v1.0.0/West_Coast/West_Coast_wave_2010.h5 . --no-sign-request
```

## Dataset Paths

### Tidal HSDS Paths

| Location | HSDS Path |
| -------- | --------- |
| Aleutian Islands | `/nlr/US_tidal/Aleutian_Islands/Aleutian_Islands_2010.h5` |
| Cook Inlet | `/nlr/US_tidal/Cook_Inlet/Cook_Inlet_2005.h5` |
| Piscataqua River | `/nlr/US_tidal/Piscataqua_River/Piscataqua_River_2007.h5` |
| Puget Sound | `/nlr/US_tidal/Puget_Sound/Puget_Sound_2015.h5` |
| Western Passage | `/nlr/US_tidal/Western_Passage/Western_Passage_2017.h5` |

### Wave HSDS Paths

| Region | HSDS Path Pattern |
| ------ | ----------------- |
| West Coast | `/nlr/US_wave/West_Coast/West_Coast_wave_{year}.h5` |
| Atlantic | `/nlr/US_wave/Atlantic/Atlantic_wave_{year}.h5` |
| Hawaii | `/nlr/US_wave/Hawaii/Hawaii_wave_{year}.h5` |

## Data Format

All files use HDF5 format (`.h5`) with:

- **Dimensions**: 2D time-series arrays (time x location)
- **Temporal axis**: Defined by `time_index` dataset
- **Spatial axis**: Defined by `coordinate` or `meta` dataset
- **Attributes**: Units, variable names, and scale factors included

For more details, see [Data Formats](data-formats.md).
