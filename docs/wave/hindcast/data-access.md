# Data Access

!!! tip "First time?"
    See [Getting Started > HSDS Setup](../../getting-started/hsds-setup.md) for installation and configuration instructions.

## Quick Start with rex

```python
from rex import ResourceX

wave_file = '/nlr/US_wave/West_Coast/West_Coast_wave_2010.h5'

with ResourceX(wave_file, hsds=True) as f:
    meta = f.meta
    time_index = f.time_index
    swh = f['significant_wave_height']
```

## Dataset Paths

### HSDS Paths

| Region | HSDS Path Pattern |
| ------ | ----------------- |
| West Coast | `/nlr/US_wave/West_Coast/West_Coast_wave_{year}.h5` |
| Atlantic | `/nlr/US_wave/Atlantic/Atlantic_wave_{year}.h5` |
| Hawaii | `/nlr/US_wave/Hawaii/Hawaii_wave_{year}.h5` |

### AWS S3 Paths

```
s3://wpto-pds-US_wave/v1.0.0/${domain}/
s3://wpto-pds-US_wave/v1.0.0/virtual_buoy/${domain}/
```

## Extract Data at a Location

```python
from rex import ResourceX

wave_file = '/nlr/US_wave/West_Coast/West_Coast_wave_2010.h5'
lat_lon = (34.399408, -119.841181)

with ResourceX(wave_file, hsds=True) as f:
    lat_lon_swh = f.get_lat_lon_df('significant_wave_height', lat_lon)
```

## Extract Data by Region

```python
from rex import ResourceX

wave_file = '/nlr/US_wave/West_Coast/West_Coast_wave_2010.h5'
jurisdiction = 'California'

with ResourceX(wave_file, hsds=True) as f:
    ca_swh = f.get_region_df('significant_wave_height', jurisdiction,
                             region_col='jurisdiction')
```

## Direct h5pyd Access

```python
import h5pyd
import pandas as pd

with h5pyd.File('/nlr/US_wave/West_Coast/West_Coast_wave_2010.h5', mode='r') as f:
    meta = pd.DataFrame(f['meta'][...])
    swh = f['significant_wave_height']
    scale_factor = swh.attrs['scale_factor']
    mean_swh = swh[...].mean(axis=0) / scale_factor

meta['Average Wave Height'] = mean_swh
```

## AWS S3 Direct Download

```bash
# List wave datasets
aws s3 ls s3://wpto-pds-US_wave/v1.0.0/ --no-sign-request

# Download a specific file
aws s3 cp s3://wpto-pds-US_wave/v1.0.0/West_Coast/West_Coast_wave_2010.h5 . --no-sign-request
```

See [AWS S3 Downloads](../../getting-started/aws-s3.md) for more details.
