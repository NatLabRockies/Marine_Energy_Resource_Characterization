# Usage Examples

Worked examples showing practical analysis workflows with the wave hindcast data.

!!! tip "Prerequisites"
    These examples assume you have completed [HSDS Setup](../../getting-started/hsds-setup.md) and have `rex` and `h5pyd` installed.

## Extract Time Series for a Single Site

```python
import h5pyd
import pandas as pd

with h5pyd.File('/nlr/US_wave/West_Coast/West_Coast_wave_2010.h5', mode='r') as f:
    time_index = pd.to_datetime(f['time_index'][...].astype(str))

    ts = pd.DataFrame(index=time_index)

    for var in ['significant_wave_height', 'mean_wave_direction', 'mean_absolute_period']:
        ds = f[var]
        scale_factor = ds.attrs['scale_factor']
        ts[var] = ds[:, 100] / scale_factor
```

## Compare Wave Resources Across Regions

```python
from rex import ResourceX

regions = {
    'West Coast': '/nlr/US_wave/West_Coast/West_Coast_wave_2010.h5',
    'Atlantic': '/nlr/US_wave/Atlantic/Atlantic_wave_2010.h5',
    'Hawaii': '/nlr/US_wave/Hawaii/Hawaii_wave_2010.h5',
}

for name, path in regions.items():
    with ResourceX(path, hsds=True) as f:
        meta = f.meta
        print(f"{name}: {len(meta)} grid points")
```

## Extract Wave Power at a Location

```python
from rex import ResourceX

wave_file = '/nlr/US_wave/West_Coast/West_Coast_wave_2010.h5'
lat_lon = (34.399408, -119.841181)  # Santa Barbara Channel

with ResourceX(wave_file, hsds=True) as f:
    power_ts = f.get_lat_lon_df('omni-directional_wave_power', lat_lon)

    print(f"Mean wave power: {power_ts.mean().values[0]:.1f} kW/m")
    print(f"Max wave power: {power_ts.max().values[0]:.1f} kW/m")
```
