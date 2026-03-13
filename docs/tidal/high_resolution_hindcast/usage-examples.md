# Usage Examples

Worked examples showing practical analysis workflows with the tidal hindcast data.

!!! tip "Prerequisites"
    These examples assume you have completed [HSDS Setup](../../getting-started/hsds-setup.md) and have `rex` and `h5pyd` installed.

## Compare Mean Current Speed Across Locations

```python
from rex import ResourceX

locations = {
    'Aleutian Islands': '/nrel/us-tidal/AK_aleutian_islands/v1.0.0/hsds/AK_aleutian_islands.wpto_high_res_tidal.hsds.v1.0.0.h5',
    'Cook Inlet': '/nrel/us-tidal/AK_cook_inlet/v1.0.0/hsds/AK_cook_inlet.wpto_high_res_tidal.hsds.v1.0.0.h5',
    'Piscataqua River': '/nrel/us-tidal/NH_piscataqua_river/v1.0.0/hsds/NH_piscataqua_river.wpto_high_res_tidal.hsds.v1.0.0.h5',
    'Puget Sound': '/nrel/us-tidal/WA_puget_sound/v1.0.0/hsds/WA_puget_sound.wpto_high_res_tidal.hsds.v1.0.0.h5',
    'Western Passage': '/nrel/us-tidal/ME_western_passage/v1.0.0/hsds/ME_western_passage.wpto_high_res_tidal.hsds.v1.0.0.h5',
}

for name, path in locations.items():
    with ResourceX(path, hsds=True) as f:
        meta = f.meta
        print(f"{name}: {len(meta)} grid points")
```

## Extract Full Time Series at a Lat/Lon

```python
from rex import ResourceX
import pandas as pd

tidal_file = '/nlr/US_tidal/Cook_Inlet/Cook_Inlet_2005.h5'
lat_lon = (60.5, -151.5)

with ResourceX(tidal_file, hsds=True) as f:
    # Extract current speed time series
    speed_ts = f.get_lat_lon_df('sea_water_speed', lat_lon)

    # Basic statistics
    print(f"Mean speed: {speed_ts.mean().values[0]:.3f} m/s")
    print(f"Max speed: {speed_ts.max().values[0]:.3f} m/s")
    print(f"P95 speed: {speed_ts.quantile(0.95).values[0]:.3f} m/s")
```

## Calculate Power Density from Speed

```python
import numpy as np

# Seawater density (nominal)
rho = 1025  # kg/m³

# Given a speed time series (m/s)
# power_density = 0.5 * rho * speed^3
power_density = 0.5 * rho * np.power(speed_ts.values, 3)  # W/m²

print(f"Mean power density: {np.mean(power_density):.1f} W/m²")
print(f"P95 power density: {np.percentile(power_density, 95):.1f} W/m²")
```

## Access Multiple Variables

```python
import h5pyd
import pandas as pd

with h5pyd.File('/nlr/US_tidal/Cook_Inlet/Cook_Inlet_2005.h5', mode='r') as f:
    time_index = pd.to_datetime(f['time_index'][...].astype(str))

    # Extract multiple variables for a single site
    site_idx = 100
    ts = pd.DataFrame(index=time_index)

    for var in ['sea_water_speed', 'sea_water_power_density']:
        ds = f[var]
```


--8<-- "docs/tidal/high_resolution_hindcast/_cite-widget.md"
