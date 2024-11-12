# High Resolution Tidal Hindcast

## Description

What is this data in two sentences

Original model output is stored in NREL managed long term storage for a period of X years

Standardized data transforms raw model output data into a common format and includes:

* Time validation and conversion to a standard format
* Coordinate validation and conversion to a standard format
* Addition of calculated engineering values specific to tidal energy development including:
    * Flow Velocity
        * Averaged and at each depth layer
    * Flow Direction
        * Averaged and at each depth layer
    * Power Density
        * Averaged and at each depth layer
    * Depth
* Organization of files my month
* Common metadata descriptions (`attrs`) in all files detailing data generation specifications,
  conventions, and other relevant information
* Unified variable is all datasets including units, name, and descriptions

The development of this dataset was funded by the U.S. Department of Energy,
Office of Energy Efficiency & Renewable Energy, Water Power Technologies Office
to improve our understanding of the U.S. tidal energy resource and to provide
critical information for tidal energy project development and tidal energy
converter design.

## Data Format

Standardized data is provided in netCDF4 format.


## Available Data

The following U.S. regions will be added to this dataset under the given `domain` names:

| Dataset                 | Duration      | Frequency        | Start                    | End                      | Count         |
| ----------------------- | ------------- | ---------------- | ------------------------ | ------------------------ | ------------- |
| `AK_Aleutian_Islands`   | 365 Days      | Hourly           | 2010-06-03 00:00         | 2011-06-02 23:30         | 8760          |
| `AK_Cook_Inlet`         | 365 Days      | Hourly           | 2005-01-01 00:00         | 2005-12-31 23:30         | 8760          |
| `ME_Western_Passage`    | 365 Days      | Half Hourly      | 2017-01-01 00:00         | 2017-12-31 23:30         | 17520         |
| `NH_Piscataqua River`   | 365 Days      | Half Hourly      | 2007-01-01 00:00         | 2007-12-31 23:30         | 17520         |
| `WA_Puget_Sound`        | 365 Days      | Half Hourly      | 2015-01-01 00:00         | 2015-12-31 23:30         | 17520         |


### Temporal Descriptions

| Dataset                 | Duration      | Frequency        | Start                    | End                      | Count         |
| ----------------------- | ------------- | ---------------- | ------------------------ | ------------------------ | ------------- |
| `AK_Aleutian_Islands`   | 365 Days      | Hourly           | 2010-06-03 00:00         | 2011-06-02 23:30         | 8760          |
| `AK_Cook_Inlet`         | 365 Days      | Hourly           | 2005-01-01 00:00         | 2005-12-31 23:30         | 8760          |
| `ME_Western_Passage`    | 365 Days      | Half Hourly      | 2017-01-01 00:00         | 2017-12-31 23:30         | 17520         |
| `NH_Piscataqua River`   | 365 Days      | Half Hourly      | 2007-01-01 00:00         | 2007-12-31 23:30         | 17520         |
| `WA_Puget_Sound`        | 365 Days      | Half Hourly      | 2015-01-01 00:00         | 2015-12-31 23:30         | 17520         |

### Location Descriptions

| Location               | Grid Type               | Avg Resolution    | Count         | Lat Min   | Lat Max   | Lon Min   | Lon Max   |
| ---------------------- | --------------------    | ----------------- | ------------- | --------- | --------- | --------- | --------- |
| `AK_Aleutian_Islands`  | Unstructured Triangular | `nan`             | `nan`         | `nan`     | `nan`     | `nan`     | `nan`     |
| `AK_Cook_Inlet`        | Unstructured Triangular | `nan`             | `nan`         | `nan`     | `nan`     | `nan`     | `nan`     |
| `ME_Western_Passages`  | Unstructured Triangular | `nan`             | `nan`         | `nan`     | `nan`     | `nan`     | `nan`     |
| `NH_Piscataqua River`  | Unstructured Triangular | `nan`             | `nan`         | `nan`     | `nan`     | `nan`     | `nan`     |
| `WA_Puget_Sound`       | Unstructured Triangular | `nan`             | `nan`         | `nan`     | `nan`     | `nan`     | `nan`     |


### Coordinate Descriptions

Coordinates define the reference system for specifying spatial and temporal positions within a dataset. In this dataset, coordinates enable selection of data points by location and time, such as extracting a time series at a specific latitude and longitude that includes velocity across all depths.


| Accessor | Name       | Units              | Description                          | Convention                                                                                              |
| -------  | ---------- | ------------------ | ------------------------------------ | ------------------------------------------------------------------------------------------------------ |
| `lat`    | Latitude   | Degrees North      | North-south position.                | [CF](https://cfconventions.org/cf-conventions/cf-conventions.html#latitude-coordinate)                 |
| `lon`    | Longitude  | Degrees East       | East-west position.                  | [CF](https://cfconventions.org/cf-conventions/cf-conventions.html#longitude-coordinate)                |
| `time`   | Time       | Seconds since 1970 | UTC time since Unix epoch.           | [CF](https://cfconventions.org/cf-conventions/cf-conventions.html#time-coordinate)                     |
| `depth`  | Depth      | Meter              | Vertical distance below the surface. | [CF](https://cfconventions.org/Data/cf-standard-names/current/build/cf-standard-name-table.html#depth) |


### Variable Descriptions

The following variables are included in each dataset:

| Accessor            | Name                                    | Units | Dimensions            | Convention                                                                                                                    |
| ---                 | ---                                     | ---   | ---                   | ---                                                                                                                           |
| `u`                 | Eastward Water Velocity                 | m s-1 | time, lat, lon, depth | [`eastward_sea_water_velocity`](https://cfconventions.org/Data/cf-standard-names/current/build/cf-standard-name-table.html#eastward_sea_water_velocity)  |
| `v`                 | Northward Water Velocity                | m s-1 | time, lat, lon, depth | [`northward_sea_water_velocity`](https://cfconventions.org/Data/cf-standard-names/current/build/cf-standard-name-table.html#northward_sea_water_velocity) |
| `ww`                | Upward Water Velocity                   | m s-1 | time, lat, lon, depth | [`upward_sea_water_velocity`](https://cfconventions.org/Data/cf-standard-names/current/build/cf-standard-name-table.html#upward_sea_water_velocity)    |
| `vap_ua`            | Depth Averaged Eastward Water Velocity  | m s-1 | time, lat, lon        | [`eastward_sea_water_velocity`](https://cfconventions.org/Data/cf-standard-names/current/build/cf-standard-name-table.html#eastward_sea_water_velocity)  |
| `vap_va`            | Depth Averaged Northward Water Velocity | m s-1 | time, lat, lon        | [`northward_sea_water_velocity`](https://cfconventions.org/Data/cf-standard-names/current/build/cf-standard-name-table.html#northward_sea_water_velocity) |
| `vap_current_speed` | Calculated Current Speed                | m s-1 | time, lat, lon, depth | [`sea_water_speed`](https://cfconventions.org/Data/cf-standard-names/current/build/cf-standard-name-table.html#sea_water_speed) |
| `vap_power_density` | Depth Averaged Northward Water Velocity | m s-1 | time, lat, lon        | [CF](https://cfconventions.org/Data/cf-standard-names/current/build/cf-standard-name-table.html#northward_sea_water_velocity) |


## Model Specifications

| Attribute                               | Description                                                                                         |
| ------------------------                | --------------------------------------------------------------------------------------------------- |
| Model                                   | [FVCOM 4.3.1](https://www.fvcom.org/)                                                                     |
| Dimensions                              | 3D                                                                                                  |
| Horizontal Resolution                   | Grid node and element points                                                                        |
| Vertical Resolution                     | 10 uniform sigma-stretched coordinates                                                              |
| Horizontal Coordinates                  | Latitude and longitude, UTM zone, State Plane                                                       |
| Vertical Datum                          | Mean Sea Level (MSL), NAVD88                                                                        |
| Temporal Resolution                     | Hourly, three hourly, daily                                                                         |
| Wetting & Drying Feature                | Activated                                                                                           |
| Boundary Forcing                        | 12 tidal constituents from OSU TPXO Tide Models                                                     |
| Wind                                    | ERA5 or CFSv2                                                                                       |
| IEC Standards                           | IEC TS 62600-201, 2015                                                                              |
| Wave-current interaction                | Not considered (small waves in study domain)                                                        |
| Atmospheric forcing                     | Not considered (wind and pressure effects negligible)                                               |
| Seawater density, salinity, temperature | Not considered (small estuarine flow in domain)                                                     |


# Accessing Data


## Python Examples

Example scripts to extract wind resource data using python are provided below:

The easiest way to access and extract data from the Resource eXtraction tool
[`rex`](https://github.com/nrel/rex)

To use `rex` with [`HSDS`](https://github.com/NREL/hsds-examples) you will need
to install `h5pyd`:

```
pip install h5pyd
```

Next you'll need to configure HSDS:

```
hsconfigure
```

and enter at the prompt:

```
hs_endpoint = https://developer.nrel.gov/api/hsds
hs_username =
hs_password =
hs_api_key = 3K3JQbjZmWctY0xmIfSYvYgtIcM3CN0cb1Y2w9bf
```

**IMPORTANT: The example API key here is for demonstation and is rate-limited per IP. To get your own API key, visit https://developer.nrel.gov/signup/**

You can also add the above contents to a configuration file at `~/.hscfg`


```python
from rex import ResourceX

wave_file = '/nrel/US_wave/West_Coast/West_Coast_wave_2010.h5'
with ResourceX(wave_file, hsds=True) as f:
    meta = f.meta
    time_index = f.time_index
    swh = f['significant_wave_height']
```

`rex` also allows easy extraction of the nearest site to a desired (lat, lon)
location:

```python
from rex import ResourceX

wave_file = '/nrel/US_wave/West_Coast/West_Coast_wave_2010.h5'
lat_lon = (34.399408, -119.841181)
with ResourceX(wave_file, hsds=True) as f:
    lat_lon_swh = f.get_lat_lon_df('significant_wave_height', lat_lon)
```

or to extract all sites in a given region:

```python
from rex import ResourceX

wave_file = '/nrel/US_wave/West_Coast/West_Coast_wave_2010.h5'
jurisdication='California'
with ResourceX(wave_file, hsds=True) as f:
    ca_swh = f.get_region_df('significant_wave_height', jurisdiction,
                             region_col='jurisdiction')
```

If you would rather access the US Wave data directly using h5pyd:

```python
# Extract the average wave height
import h5pyd
import pandas as pd

# Open .h5 file
with h5pyd.File('/nrel/US_wave/West_Coast/West_Coast_wave_2010.h5', mode='r') as f:
    # Extract meta data and convert from records array to DataFrame
    meta = pd.DataFrame(f['meta'][...])
    # Significant Wave Height
    swh = f['significant_wave_height']
    # Extract scale factor
    scale_factor = swh.attrs['scale_factor']
    # Extract, average, and unscale wave height
    mean_swh = swh[...].mean(axis=0) / scale_factor

# Add mean wave height to meta data
meta['Average Wave Height'] = mean_swh
```

```python
# Extract time-series data for a single site
import h5pyd
import pandas as pd

# Open .h5 file
with h5pyd.File('/nrel/US_wave/West_Coast/West_Coast_wave_2010.h5', mode='r') as f:
    # Extract time_index and convert to datetime
    # NOTE: time_index is saved as byte-strings and must be decoded
    time_index = pd.to_datetime(f['time_index'][...].astype(str))
    # Initialize DataFrame to store time-series data
    time_series = pd.DataFrame(index=time_index)
    # Extract wave height, direction, and period
    for var in ['significant_wave_height', 'mean_wave_direction',
                'mean_absolute_period']:
    	# Get dataset
    	ds = f[var]
    	# Extract scale factor
    	scale_factor = ds.attrs['scale_factor']
    	# Extract site 100 and add to DataFrame
    	time_series[var] = ds[:, 100] / scale_factor
```
## References

Please cite the most relevant publication below when referencing this dataset:

1) [Wu, Wei-Cheng, et al. "Development and validation of a high-resolution regional wave hindcast model for US West Coast wave resource characterization." Renewable Energy 152 (2020): 736-753.](https://www.osti.gov/biblio/1599105)
2) [Yang, Z., G. García-Medina, W. Wu, and T. Wang, 2020. Characteristics and variability of the Nearshore Wave Resource on the U.S. West Coast. Energy.](https://doi.org/10.1016/j.energy.2020.117818)
3) [Yang, Zhaoqing, et al. High-Resolution Regional Wave Hindcast for the US West Coast. No. PNNL-28107. Pacific Northwest National Lab.(PNNL), Richland, WA (United States), 2018.](https://doi.org/10.2172/1573061)
4) [Ahn, S. V.S. Neary, Allahdadi, N. and R. He, Nearshore wave energy resource characterization along the East Coast of the United States, Renewable Energy, 2021, 172](https://doi.org/10.1016/j.renene.2021.03.037)
5) [Yang, Z. and V.S. Neary, High-resolution hindcasts for U.S. wave energy resource characterization. International Marine Energy Journal, 2020, 3, 65-71](https://doi.org/10.36688/imej.3.65-71)
6) [Allahdadi, M.N., He, R., and Neary, V.S.: Predicting ocean waves along the US East Coast during energetic winter storms: sensitivity to whitecapping parameterizations, Ocean Sci., 2019, 15, 691-715](https://doi.org/10.5194/os-15-691-2019)
7) [Allahdadi, M.N., Gunawan, J. Lai, R. He, V.S. Neary, Development and validation of a regional-scale high-resolution unstructured model for wave energy resource characterization along the US East Coast, Renewable Energy, 2019, 136, 500-511](https://doi.org/10.1016/j.renene.2019.01.020)

## Disclaimer and Attribution

The National Renewable Energy Laboratory (“NREL”) is operated for the U.S.
Department of Energy (“DOE”) by the Alliance for Sustainable Energy, LLC
("Alliance"). Pacific Northwest National Laboratory (PNNL) is managed and
operated by Battelle Memorial Institute ("Battelle") for DOE. As such the
following rules apply:

This data arose from worked performed under funding provided by the United
States Government. Access to or use of this data ("Data") denotes consent with
the fact that this data is provided "AS IS," “WHEREIS” AND SPECIFICALLY FREE
FROM ANY EXPRESS OR IMPLIED WARRANTY OF ANY KIND, INCLUDING BUT NOT LIMITED TO
ANY IMPLIED WARRANTIES SUCH AS MERCHANTABILITY AND/OR FITNESS FOR ANY
PARTICULAR PURPOSE. Furthermore, NEITHER THE UNITED STATES GOVERNMENT NOR ANY
OF ITS ASSOCITED ENTITES OR CONTRACTORS INCLUDING BUT NOT LIMITED TO THE
DOE/PNNL/NREL/BATTELLE/ALLIANCE ASSUME ANY LEGAL LIABILITY OR RESPONSIBILITY
FOR THE ACCURACY, COMPLETENESS, OR USEFULNESS OF THE DATA, OR REPRESENT THAT
ITS USE WOULD NOT INFRINGE PRIVATELY OWNED RIGHTS. NO ENDORSEMENT OF THE DATA
OR ANY REPRESENTATIONS MADE IN CONNECTION WITH THE DATA IS PROVIDED. IN NO
EVENT SHALL ANY PARTY BE LIABLE FOR ANY DAMAGES, INCLUDING BUT NOT LIMITED TO
SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES ARISING FROM THE PROVISION OF THIS
DATA; TO THE EXTENT PERMITTED BY LAW USER AGREES TO INDEMNIFY
DOE/PNNL/NREL/BATTELLE/ALLIANCE AND ITS SUBSIDIARIES, AFFILIATES, OFFICERS,
AGENTS, AND EMPLOYEES AGAINST ANY CLAIM OR DEMAND RELATED TO USER'S USE OF THE
DATA, INCLUDING ANY REASONABLE ATTORNEYS FEES INCURRED.

The user is granted the right, without any fee or cost, to use or copy the
Data, provided that this entire notice appears in all copies of the Data. In
the event that user engages in any scientific or technical publication
utilizing this data user agrees to credit DOE/PNNL/NREL/BATTELLE/ALLIANCE in
any such publication consistent with respective professional practice.
