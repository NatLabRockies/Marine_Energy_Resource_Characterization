# Data Access

Dataset: [@mhkdr_submission]

!!! tip "First time?"
    See [Getting Started > HSDS Setup](../../getting-started/hsds-setup.md) for installation and configuration instructions.

## Quick Start

```python
from rex import ResourceX

# Access Cook Inlet tidal data via HSDS
tidal_file = '/nlr/US_tidal/Cook_Inlet/Cook_Inlet_2005.h5'

with ResourceX(tidal_file, hsds=True) as f:
    meta = f.meta
    time_index = f.time_index
    speed = f['sea_water_speed']
```

## Dataset Paths

| Location | HSDS Path |
| -------- | --------- |
| Aleutian Islands | `/nlr/US_tidal/Aleutian_Islands/Aleutian_Islands_2010.h5` |
| Cook Inlet | `/nlr/US_tidal/Cook_Inlet/Cook_Inlet_2005.h5` |
| Piscataqua River | `/nlr/US_tidal/Piscataqua_River/Piscataqua_River_2007.h5` |
| Puget Sound | `/nlr/US_tidal/Puget_Sound/Puget_Sound_2015.h5` |
| Western Passage | `/nlr/US_tidal/Western_Passage/Western_Passage_2017.h5` |

## Python Examples with rex

### Access Metadata

```python
from rex import ResourceX

tidal_file = '/nlr/US_tidal/Cook_Inlet/Cook_Inlet_2005.h5'

with ResourceX(tidal_file, hsds=True) as f:
    meta = f.meta          # DataFrame with lat, lon, and spatial metadata
    time_index = f.time_index  # DatetimeIndex of all timesteps
    print(f"Grid points: {len(meta)}")
    print(f"Timesteps: {len(time_index)}")
```

### Extract Time Series at a Location

```python
from rex import ResourceX

tidal_file = '/nlr/US_tidal/Cook_Inlet/Cook_Inlet_2005.h5'
lat_lon = (60.5, -151.5)  # Cook Inlet location

with ResourceX(tidal_file, hsds=True) as f:
    speed_ts = f.get_lat_lon_df('sea_water_speed', lat_lon)
```

## Python Examples with h5pyd

### Direct Data Access

```python
import h5pyd
import pandas as pd

with h5pyd.File('/nlr/US_tidal/Cook_Inlet/Cook_Inlet_2005.h5', mode='r') as f:
    meta = pd.DataFrame(f['meta'][...])
    speed = f['sea_water_speed']
    scale_factor = speed.attrs['scale_factor']
    mean_speed = speed[...].mean(axis=0) / scale_factor

meta['Mean Speed'] = mean_speed
```

## AWS S3 Direct Download

For bulk data access without HSDS:

```bash
# Download a specific file
aws s3 cp s3://marine-energy-data/US_tidal/Cook_Inlet/Cook_Inlet_2005.h5 . --no-sign-request

# Download all files for a location
aws s3 sync s3://marine-energy-data/US_tidal/Cook_Inlet/ ./Cook_Inlet/ --no-sign-request
```

See [AWS S3 Downloads](../../getting-started/aws-s3.md) for more details.


--8<-- "docs/tidal/high_resolution_hindcast/_cite-widget.md"
