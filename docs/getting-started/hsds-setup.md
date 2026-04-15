# HSDS Setup

The Highly Scalable Data Service (HSDS) provides programmatic access to all marine energy datasets without downloading files.

## Prerequisites

- Python 3.8+
- pip

## Installation

Install the required packages:

```bash
pip install rex h5pyd
```

## Configuration

Run the HSDS configuration utility:

```bash
hsconfigure
```

Enter the following at the prompt:

```
hs_endpoint = https://developer.nlr.gov/api/hsds
hs_username =
hs_password =
hs_api_key = YOUR_API_KEY
```

!!! warning "API Key Required"
    Get your own API key at [https://developer.nlr.gov/signup/](https://developer.nlr.gov/signup/). The example key is rate-limited per IP.

## Verify Connection

Test your setup with a quick data access:

```python
from rex import ResourceX

# Test with tidal data
tidal_file = '/nlr/US_tidal/Cook_Inlet/Cook_Inlet_2005.h5'

with ResourceX(tidal_file, hsds=True) as f:
    print(f"Tidal meta shape: {f.meta.shape}")
    print(f"Time index length: {len(f.time_index)}")
```

```python
# Test with wave data
wave_file = '/nlr/US_wave/West_Coast/West_Coast_wave_2010.h5'

with ResourceX(wave_file, hsds=True) as f:
    print(f"Wave meta shape: {f.meta.shape}")
    print(f"Time index length: {len(f.time_index)}")
```

## Direct h5pyd Access

For lower-level access without `rex`:

```python
import h5pyd
import pandas as pd

with h5pyd.File('/nlr/US_tidal/Cook_Inlet/Cook_Inlet_2005.h5', mode='r') as f:
    meta = pd.DataFrame(f['meta'][...])
    speed = f['sea_water_speed']
    scale_factor = speed.attrs['scale_factor']
    mean_speed = speed[...].mean(axis=0) / scale_factor
```

## HSDS Endpoints

| Dataset | HSDS Path |
| ------- | --------- |
| Tidal   | `/nlr/US_tidal/` |
| Wave    | `/nlr/US_wave/` |

## Troubleshooting

**"403 Forbidden" error**

Your API key may be invalid or expired. Regenerate it at [developer.nlr.gov/signup](https://developer.nlr.gov/signup/).

**"Connection refused" error**

Check that your `~/.hscfg` file has the correct endpoint: `https://developer.nlr.gov/api/hsds`

**Slow data access**

HSDS streams data over the network. For bulk downloads, consider using [AWS S3](aws-s3.md) instead.

For additional examples, see the [HSDS Examples Repository](https://github.com/nlr/hsds-examples).
