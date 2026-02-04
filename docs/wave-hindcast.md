# High Resolution Ocean Surface Wave Hindcast

32-year high-resolution wave hindcast covering the U.S. Exclusive Economic Zone, generated using WaveWatch III and SWAN models.

## Overview

The development of this dataset was funded by the U.S. Department of Energy, Office of Energy Efficiency & Renewable Energy, Water Power Technologies Office to improve our understanding of the U.S. wave energy resource and to provide critical information for wave energy project development and wave energy converter design.

This is the highest resolution publicly available long-term wave hindcast dataset covering the entire U.S. Exclusive Economic Zone (EEZ). The data can be used to investigate the historical record of wave statistics at any U.S. site and could be of value to any entity with marine operations inside the U.S. EEZ.

!!! info "Dataset Summary" - **Duration**: 32-Year Wave Hindcast (1979-2010) - **Temporal Resolution**: 3-hour intervals - **Spatial Resolution**: 200 meters (shallow water) to ~10 km (deep water) - **Coverage**: U.S. Exclusive Economic Zone

## Regional Coverage

| Region                                           | Domain Name  | Status      |
| ------------------------------------------------ | ------------ | ----------- |
| West Coast United States                         | `West_Coast` | Available   |
| East Coast United States                         | `Atlantic`   | Available   |
| Hawaiian Islands                                 | `Hawaii`     | Available   |
| Alaskan Coast                                    | TBD          | Coming Soon |
| Gulf of Mexico, Puerto Rico, U.S. Virgin Islands | TBD          | Coming Soon |
| U.S. Pacific Island Territories                  | TBD          | Coming Soon |

## Available Variables

| Variable                    | Description                                | Units   |
| --------------------------- | ------------------------------------------ | ------- |
| Mean Wave Direction         | Direction normal to wave crests            | degrees |
| Significant Wave Height     | Zeroth spectral moment (H_m0)              | m       |
| Mean Absolute Period        | Ratio of spectral moments (m_0/m_1)        | s       |
| Peak Period                 | Period of maximum wave energy              | s       |
| Mean Zero-Crossing Period   | Ratio of spectral moments (√(m_0/m_2))     | s       |
| Energy Period               | Ratio of spectral moments (m\_-1/m_0)      | s       |
| Directionality Coefficient  | Fraction of energy in max power direction  | -       |
| Maximum Energy Direction    | Direction of maximum wave power            | degrees |
| Omni-Directional Wave Power | Total wave energy flux from all directions | kW/m    |
| Spectral Width              | Relative spreading of energy in spectrum   | -       |

## Model Description

The multi-scale, unstructured-grid modeling approach using WaveWatch III and SWAN enabled long-term (decades) high-resolution hindcasts in a large regional domain:

- **Outer Model**: WaveWatch III with global-regional nested grids
- **Inner Model**: Unstructured-grid SWAN with resolution as fine as 200 meters in shallow waters
- **Timestep**: 3-hour intervals spanning 32 years (1979-2010)

The models were extensively validated against:

- Common wave parameters
- Six IEC resource parameters
- 2D spectra from high-quality spectral buoy data

!!! note "Future Extension"
The project team intends to extend this dataset to 2020 (i.e., 1979-2020), pending DOE support.

## Data Access

### AWS S3 Storage

High Resolution Ocean Surface Wave Hindcast data is available on AWS S3:

```
s3://wpto-pds-US_wave/v1.0.0/${domain}
```

Virtual buoy hourly data:

```
s3://wpto-pds-US_wave/v1.0.0/virtual_buoy/${domain}
```

### HSDS Access

The US wave data is available via HSDS at `/nlr/US_wave/`

For examples on setting up and using HSDS, see the [HSDS Examples Repository](https://github.com/nlr/hsds-examples).

## Data Format

The data is provided in high-density HDF5 files (.h5) separated by year:

- **Dimensions**: 2D time-series arrays (time × location)
- **Temporal Axis**: Defined by `time_index` dataset
- **Spatial Axis**: Defined by `coordinate` dataset
- **Attributes**: Units, SWAN names, IEC names included

## Python Examples

### Using rex (Recommended)

Install rex and h5pyd:

```bash
pip install rex h5pyd
```

Configure HSDS:

```bash
hsconfigure
```

Enter at the prompt:

```
hs_endpoint = https://developer.nlr.gov/api/hsds
hs_username =
hs_password =
hs_api_key = YOUR_API_KEY
```

!!! warning "API Key Required"
Get your own API key at [https://developer.nlr.gov/signup/](https://developer.nlr.gov/signup/). The example key is rate-limited per IP.

### Basic Data Access

```python
from rex import ResourceX

wave_file = '/nlr/US_wave/West_Coast/West_Coast_wave_2010.h5'

with ResourceX(wave_file, hsds=True) as f:
    meta = f.meta
    time_index = f.time_index
    swh = f['significant_wave_height']
```

### Extract Data at a Location

```python
from rex import ResourceX

wave_file = '/nlr/US_wave/West_Coast/West_Coast_wave_2010.h5'
lat_lon = (34.399408, -119.841181)

with ResourceX(wave_file, hsds=True) as f:
    lat_lon_swh = f.get_lat_lon_df('significant_wave_height', lat_lon)
```

### Extract Data by Region

```python
from rex import ResourceX

wave_file = '/nlr/US_wave/West_Coast/West_Coast_wave_2010.h5'
jurisdiction = 'California'

with ResourceX(wave_file, hsds=True) as f:
    ca_swh = f.get_region_df('significant_wave_height', jurisdiction,
                             region_col='jurisdiction')
```

### Direct h5pyd Access

```python
import h5pyd
import pandas as pd

with h5pyd.File('/nlr/US_wave/West_Coast/West_Coast_wave_2010.h5', mode='r') as f:
    # Extract metadata
    meta = pd.DataFrame(f['meta'][...])

    # Significant Wave Height
    swh = f['significant_wave_height']
    scale_factor = swh.attrs['scale_factor']

    # Calculate mean wave height
    mean_swh = swh[...].mean(axis=0) / scale_factor

meta['Average Wave Height'] = mean_swh
```

### Extract Time Series for a Single Site

```python
import h5pyd
import pandas as pd

with h5pyd.File('/nlr/US_wave/West_Coast/West_Coast_wave_2010.h5', mode='r') as f:
    # Extract time index
    time_index = pd.to_datetime(f['time_index'][...].astype(str))

    # Initialize DataFrame
    time_series = pd.DataFrame(index=time_index)

    # Extract multiple variables for site 100
    for var in ['significant_wave_height', 'mean_wave_direction', 'mean_absolute_period']:
        ds = f[var]
        scale_factor = ds.attrs['scale_factor']
        time_series[var] = ds[:, 100] / scale_factor
```

## References

Please cite the most relevant publication when referencing this dataset:

1. Wu, Wei-Cheng, et al. ["Development and validation of a high-resolution regional wave hindcast model for US West Coast wave resource characterization."](https://www.osti.gov/biblio/1599105) Renewable Energy 152 (2020): 736-753.

2. Yang, Z., G. García-Medina, W. Wu, and T. Wang, 2020. ["Characteristics and variability of the Nearshore Wave Resource on the U.S. West Coast."](https://doi.org/10.1016/j.energy.2020.117818) Energy.

3. Yang, Zhaoqing, et al. ["High-Resolution Regional Wave Hindcast for the US West Coast."](https://doi.org/10.2172/1573061) No. PNNL-28107. Pacific Northwest National Lab.(PNNL), Richland, WA (United States), 2018.

4. Ahn, S., V.S. Neary, Allahdadi, N. and R. He, ["Nearshore wave energy resource characterization along the East Coast of the United States."](https://doi.org/10.1016/j.renene.2021.03.037) Renewable Energy, 2021, 172.

5. Yang, Z. and V.S. Neary, ["High-resolution hindcasts for U.S. wave energy resource characterization."](https://doi.org/10.36688/imej.3.65-71) International Marine Energy Journal, 2020, 3, 65-71.

6. Allahdadi, M.N., He, R., and Neary, V.S.: ["Predicting ocean waves along the US East Coast during energetic winter storms: sensitivity to whitecapping parameterizations."](https://doi.org/10.5194/os-15-691-2019) Ocean Sci., 2019, 15, 691-715.

7. Allahdadi, M.N., Gunawan, J. Lai, R. He, V.S. Neary, ["Development and validation of a regional-scale high-resolution unstructured model for wave energy resource characterization along the US East Coast."](https://doi.org/10.1016/j.renene.2019.01.020) Renewable Energy, 2019, 136, 500-511.

## Acknowledgement

This study was funded by the U.S. Department of Energy, Office of Energy Efficiency & Renewable Energy, Water Power Technologies Office under Contract DE-AC05-76RL01830 to Pacific Northwest National Laboratory (PNNL).

## Disclaimer and Attribution

The National Laboratory of the Rockies ("NLR") is operated for the U.S. Department of Energy ("DOE") by the Alliance for Energy Innovation, LLC ("Alliance"). Pacific Northwest National Laboratory (PNNL) is managed and operated by Battelle Memorial Institute ("Battelle") for DOE.

This data arose from work performed under funding provided by the United States Government. Access to or use of this data ("Data") denotes consent with the fact that this data is provided "AS IS," "WHERE IS" and specifically free from any express or implied warranty of any kind, including but not limited to any implied warranties such as merchantability and/or fitness for any particular purpose.

Furthermore, neither the United States Government nor any of its associated entities or contractors including but not limited to DOE/PNNL/NLR/BATTELLE/ALLIANCE assume any legal liability or responsibility for the accuracy, completeness, or usefulness of the data, or represent that its use would not infringe privately owned rights.

The user is granted the right, without any fee or cost, to use or copy the Data, provided that this entire notice appears in all copies of the Data. In the event that user engages in any scientific or technical publication utilizing this data, user agrees to credit DOE/PNNL/NLR/BATTELLE/ALLIANCE in any such publication consistent with respective professional practice.
