# Getting Started

## What's Available

The Marine Energy Resource Characterization project provides two categories of publicly accessible datasets:

| Dataset | Energy Type | Coverage | Duration | Resolution |
| ------- | ----------- | -------- | -------- | ---------- |
| [High Resolution Tidal Hindcast](../tidal/high_resolution_hindcast/index.md) | Tidal | 5 U.S. coastal locations | 1 year each | Hourly / Half-hourly |
| [Ocean Surface Wave Hindcast](../wave/hindcast/index.md) | Wave | U.S. Exclusive Economic Zone | 32 years (1979-2010) | 3-hourly |

## Quick Start

Access tidal data in 3 lines of Python:

```python
from rex import ResourceX

tidal_file = '/nlr/US_tidal/Cook_Inlet/Cook_Inlet_2005.h5'

with ResourceX(tidal_file, hsds=True) as f:
    meta = f.meta
    time_index = f.time_index
    speed = f['sea_water_speed']
```

!!! tip "Prerequisites"
    You'll need `rex` and `h5pyd` installed and configured. See [HSDS Setup](hsds-setup.md) for step-by-step instructions.

Access wave data:

```python
from rex import ResourceX

wave_file = '/nlr/US_wave/West_Coast/West_Coast_wave_2010.h5'

with ResourceX(wave_file, hsds=True) as f:
    meta = f.meta
    swh = f['significant_wave_height']
```

## Next Steps

- **[HSDS Setup](hsds-setup.md)** — Configure programmatic data access (required for code examples)
- **[AWS S3 Downloads](aws-s3.md)** — Browse and download data directly
- **[Marine Energy Atlas](marine-energy-atlas.md)** — Visualize summary data interactively
- **[Tidal Dataset Documentation](../tidal/high_resolution_hindcast/index.md)** — Full tidal hindcast documentation
- **[Wave Dataset Documentation](../wave/hindcast/index.md)** — Full wave hindcast documentation
