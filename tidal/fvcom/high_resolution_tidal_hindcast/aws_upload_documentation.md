# AWS Open Data Registry Questionnaire


<!--
&#10;AWS questionnaire:
• Contact name.
• Contact email address and link to GitHub issues page (if relevant).
• Name of institution.
• Website of institution.
• Name of dataset.
• Brief description of dataset (about 200 words to describe the data, how it is created, and what it is used for).
• Size of dataset (e.g. 100GB).
• Growth rate of dataset in GB per year, if applicable (e.g. 50GB per year).
• Dataset file formats (e.g. Parquet, ORC, csv, cloud-optimized GeoTiff, NetCDF, json).
• Dataset license (e.g. Creative Commons, no license, US Government work).
• Explain how will this data could be used (e.g. training machine learning models, climate research, computer vision, energy efficiency, genome analysis, training, agricultural research, demo development. Specific tools or workflows would be very helpful).
• What’s new or novel about this dataset or the way it is provided? (Tell us why this dataset is awesome).
• How dataset has been optimized to be used in the cloud.
• Other DOE labs, institutions or companies already using this data.
• URLs pointing to documentation of the structure and content of the dataset (attaching a readme file is acceptable if no such link exists).
• Do you have any tutorials or example data usage materials (i.e. Jupyter Notebook) that can be modified to access the data via AWS? If so, please include a URL or attach a file.
• How NREL intends to promote the existence of this dataset on AWS, including blogs, mailing lists, or other channels.
• Tags (chosen from this list: https://github.com/awslabs/open-data-registry/blob/main/tags.yaml. If you’d like to add tags that aren’t on this list, notate them as “not in list”).
• URLs to any papers or reports about this dataset.
&#10;-->

# Contact Information

|  |  |
|----|----|
| **Contact Name** | Michael Lawson (NREL PI) |
| **Contact Email** | <michael.lawson@nrel.gov> |
| **Reporting Issues** | [US Tidal Dataset Issue Reporting](https://github.com/NREL/Marine_Energy_Resource_Characterization/issues) |
| **Institution Names** | National Renewable Energy Laboratory (NREL), Pacific Northwest National Laboratory (PNNL) , Water Power Technologies Office (WPTO) |
| **Institution Websites** | [NREL Marine Energy Resource Characterizaion](https://www.nrel.gov/water/resource-characterization), [PNNL Coastal Sciences Division](https://www.pnnl.gov/coastal-sciences-division), [WPTO Marine Energy Resource Assessment and Characterization](https://www.energy.gov/eere/water/marine-energy-resource-assessment-and-characterization) |
| **MHKDR Submission** | [High Resolution Tidal Hindcast (US Tidal)](https://mhkdr.openei.org/submissions/632) |

# Dataset Information

## Dataset Name

WPTO High Resolution Tidal Hindcast

<!--
- **Brief description** (~200 words):
-->

## Description

The WPTO High Resolution Tidal Hindcast dataset provides standardized
tidal energy data for five strategically selected U.S. coastal locations
with significant tidal energy potential. Developed collaboratively by
Pacific Northwest National Laboratory (PNNL) and National Renewable
Energy Laboratory (NREL), this dataset is funded by the U.S. Department
of Energy’s Water Power Technologies Office Marine Energy Resource
Assessment and Characterization project.

Generated using FVCOM 4.3.1 (Finite Volume Community Ocean Model) \[1\],
the dataset contains one year of high-resolution tidal energy data
including eastward and northward sea water velocities, surface
elevation, calculated speed, flow direction, and power density across 10
uniform sigma layers. The data follows
[CF-1.10](https://cfconventions.org/),
[ACDD-1.3](https://wiki.esipfed.org/ACDD), and [ME Data
Pipeline-1.0](https://github.com/tsdat/data_standards/blob/main/ME_DataStandards.pdf)
conventions and meets IEC 62600-201 \[2\] Stage 1 tidal resource
analysis standards.

The five locations include Aleutian Islands and Cook Inlet (Alaska),
Piscataqua River (New Hampshire), Puget Sound (Washington), and Western
Passage (Maine), with temporal resolutions ranging from half-hourly to
hourly and spanning complete annual cycles for each dataset. This
standardized dataset supports theoretical and technical resource
potential assessments, commercial development planning, policy analysis,
environmental planning, and research applications for the marine energy
community.

- **Dataset size**: ~10TB (combined all locations)

- **Growth rate**: Static dataset - no planned annual growth. Future
  expansions may add new coastal locations.

- **File formats**: NetCDF4 (\*.nc), Apache Parquet (.parquet), HDF5
  (.h5)

- **License**: US Government work (Freely Distributed)

## Dataset Specifications

### Locations

1.  **Aleutian Islands, Alaska** (1 Year, 2010-2011, hourly, 797,978
    grid faces)
2.  **Cook Inlet, Alaska** (1 Year, 2005, hourly, 392,002 grid faces)
3.  **Piscataqua River, New Hampshire** (1 Year, 2007, half-hourly,
    292,927 grid faces)
4.  **Puget Sound, Washington** (364 Days, 2015, half-hourly, 1,734,765
    grid faces)
5.  **Western Passage, Maine** (1 Year, 2017, half-hourly, 231,208 grid
    faces)

### Variables

- Latitude and longitude \[degrees\]
- Eastward/Northward sea water velocity (u, v) \[meters per second\]
- Depth and Surface Elevation calculated to NAVD88 \[meters\]
- Calculated sea water speed \[meters per second\]
- Calculated sea water to direction \[degrees clockwise from true
  north\]
- Calculated sea water power density \[watts per square meter\]
- Calculated surface elevation relative to Mean Sea Level (MSL)
  \[meters\]
- Calculated total depth and layer depths relative to MSL \[meters\]

### Model Configuration

- FVCOM 4.3.1 \[1\] with unstructured triangular mesh
- 10 uniform sigma (depth) layers from surface to seafloor
- Sub-500m grid resolution in many tidal energy areas
- 12 tidal constituents from [OSU TPXO](https://www.tpxo.net/) boundary
  forcing
- [ERA5](https://www.ecmwf.int/en/forecasts/dataset/ecmwf-reanalysis-v5)/[CFSv2](https://www.cpc.ncep.noaa.gov/products/CFSv2/CFSv2_body.html)
  atmospheric forcing for wind effects

### Usage and Applications

- Stage 1 tidal energy resource assessment and site characterization
  following the IEC 62600-201 \[2\] tidal energy resource assessment
  standard
- Training machine learning models for tidal energy prediction and
  optimization
- Climate and oceanographic research studying tidal dynamics and coastal
  processes
- Device design and engineering analysis for tidal energy converters
- Environmental impact assessment and marine spatial planning
- Commercial tidal energy project development and feasibility studies
- Academic research and education in marine renewable energy
- Policy analysis and regulatory planning for marine energy development
- Integration with existing energy system models and grid planning tools

### Novel Aspects

- First standardized, publicly accessible high-resolution tidal hindcast
  dataset following IEC tidal energy resource assessment standards \[2\]
- Comprehensive 3D water column characterization with 10 dynamically
  adjusting sigma layers that respond to tidal elevation changes
- Multi-laboratory collaborative effort ensuring rigorous quality
  control and validation against ADCP measurements
- Transparent, open-source processing workflow available on GitHub for
  full reproducibility
- Unstructured triangular mesh topology optimized for complex coastal
  geometries with sub-500m resolution in most tidal energy areas
- Integration of 12 tidal constituents from [OSU
  TPXO](https://www.tpxo.net/) models with
  [ERA5](https://www.ecmwf.int/en/forecasts/dataset/ecmwf-reanalysis-v5)/[CFSv2](https://www.cpc.ncep.noaa.gov/products/CFSv2/CFSv2_body.html)
  atmospheric forcing
- Value-added engineering products including power density calculations
  using marine energy industry standards
- Multiple data formats for both detailed site-specific analysis and
  large-scale resource assessment

<!--
&#10;**How dataset has been optimized for cloud use**:
&#10;-->

### Cloud Optimization

- Multiple data products tailored for different analysis needs built
  from standardized original data:
  - Yearly Individual Point data files for site-specific analysis
    - Partitioned Apache Parquet (`.parquet`) files by
      latitude/longitude for efficient access of individual yearly point
      data
    - JSON partition manifest with accompanying Python library to
      facilitate data access, analysis and visualization at individual
      points
  - Complete spatial datasets in monthly files for temporal large area
    resource assessment
    - Standardized NetCDF4 (`.nc`) data products for spatiotemponal
      analysis and visualization
  - Complete yearly datasets for HSDS/Cloud optimized access, utilized
    by the Marine Energy Atlas to serve data for spatial queries
    - HDF5 (`.h5`) data products are uncompressed and optimized for
      chunked access and efficient subsetting using
  - Yearly summary outputs in NetCDF, Parquet formats and GIS (GeoJson,
    GPKG and GeoParquet) for summary analysis.
- All data products contain additional value-added engineering variables
  (speed, direction, depth, surface elevation) for tidal energy analysis
- All products contain additional column and global metadata to provide
  context for analysis.

<!--
**DOE labs, institutions, or companies using this data**:
-->

### Current Users

Data is currently in pre-release stage and not yet widely adopted.
Expected usage includes tidal energy resource characterization, and
tidal energy techno-economic analysis.

<!--
&#10;**URLs to dataset documentation**:
&#10;-->

### Documentation

- [Marine Energy Resource Characterization Software
  Development](https://github.com/NREL/Marine_Energy_Resource_Characterization)
- [Marine Energy Atlas Dataset
  Visualization](https://maps.nrel.gov/marine-energy-atlas/data-viewer/data-library/layers)
- [High Resolution Tidal Hindcast Processing
  Software](https://github.com/NREL/Marine_Energy_Resource_Characterization/tidal/fvcom/high_resolution_tidal_hindcast)
  (repository includes detailed methodology documentation)

<!--
&#10;**Tutorials or example usage materials**:
&#10;Basic data access and visualization example:
&#10;-->

# Example

WPTO Tidal Hindcast dataset is available in multiple data products,
including point data for the duration of the model run. The python
library `marine_energy_hindcast` provides a simple interface to download
the data at a single point including the complete data for the 1 year
model run, metadata included in parquet files hosted by the AWS Open
Data Registry [US Tidal](https://registry.opendata.aws/us-tidal/). To
profivide additioval value, analysis and visualization functions are
included in the library include IEC 62600-201 \[2\] compliant joint
probability distribution, probability of exceedance calculations and
additional visualizations relevant to tidal energy resource assessment.

### Pre-requisites

- Python 3.10+
- Internet access to download data from AWS Open Data Registry

``` bash
pip install marine_energy_hindcast
```

### Download and Query Complete Data at Single Point

This example demonstrates how to download the complete tidal hindcast
data for a specific point. Users are expected to visit the Marine Energy
Atlas [Data
Library](https://maps.nrel.gov/marine-energy-atlas/data-viewer/data-library/layers)
to determine points of interest. The example below uses a point in Cook
Inlet, Alaska, near Nikiski. Which is a known tidal energy site with
significant potential for tidal energy development.

First the library is imported and the target coordinates are defined.
The `get_data_at_point` function is then used to download the data and
metadata for the specified point.

``` python
import marine_energy_hindcast.tidal as tidal_hindcast

# Target coordinates
# Cook Inlet, Near Nikiski, AK
lat = 60.750500
lon = -151.446533

df, metadata = tidal_hindcast.query_nearest_point(
    lat,
    lon,
    return_metadata=True
)
```

The `query_nearest_point` function returns a pandas DataFrame containing
the complete tidal hindcast data for the specified point, and a metadata
dictionary with information about the dataset. There are many different
ways to use this data. To start we will create visualizations specified
by the Tidal Energy Resource Assessment Standard IEC 62600-201 \[2\].

### Visualizing Tidal Data

The `marine_energy_hindcast.tidal.viz` module includes functions to
create common visualizations used in tidal energy resource assessment.
The following examples demonstrate how to create a joint probability
distribution plot, probability of exceedance plot, and time series plots
of speed, to direction, and surface elevation.

``` python
tidal_hindcast.viz.joint_probability_distribution(df, metadata, sigma_layer=1)
```

![Joint Probability Distribution
Example](./docs/img/py_output/ak_cook_inlet.wpto_high_res_tidal_hindcast.lat=60.750500.lon=-151.446533.tidal_joint_probability_distribution_sigma_layer_1_depth_range_5.44_to_6.75_m.png)

``` python
tidal_hindcast.viz.probability_of_exceedance(df, metadata)
```

![Velocity Exceedance
Example](./docs/img/py_output/ak_cook_inlet.wpto_high_res_tidal_hindcast.lat=60.750500.lon=-151.446533.velocity_exceedance_probability.png)

``` python
tidal_hindcast.viz.speed(df, metadata, start_date="2005-01-01", number_of_days=7)
```

![Speed Over Time
Example](./docs/img/py_output/ak_cook_inlet.wpto_high_res_tidal_hindcast.lat=60.732075.lon=-151.431580_7days_2005-01-01.sea_water_speed.png)

``` python
tidal_hindcast.viz.to_direction(df, metadata, start_date="2005-01-01", number_of_days=7)
```

![To Direction Over Time
Example](./docs/img/py_output/ak_cook_inlet.wpto_high_res_tidal_hindcast.lat=60.732075.lon=-151.431580_7days_2005-01-01.sea_water_to_direction.png)

``` python
tidal_hindcast.viz.surface_elevation(df, metadata, start_date="2005-01-01", number_of_days=7)
```

![Surface Elevation Over Time
Example](./docs/img/py_output/ak_cook_inlet.wpto_high_res_tidal_hindcast.lat=60.732075.lon=-151.431580_7days_2005-01-01.surface_elevation_analysis.png)

Additionally a full overview of the data can be created using the
`overview` function:

``` python
tidal_hindcast.viz.overview(df, metadata)
```

![Point Data Overview
Example](./docs/img/py_output/ak_cook_inlet.wpto_high_res_tidal_hindcast.lat=60.750500.lon=-151.446533.velocity_and_direction_overview.png)

Additional data resulting dataframe contains the complete tidal hindcast
data for the specified point as a pandas DataFrame, and the metadata is
returned as a dictionary. In the original data from FVCOM model the `u`
and `v` vectors are calculated at 10 uniform sigma layers from the
surface to the seafloor. This dataset includes additional calculated
variables that are prefixed with `vap_`, which stands for “value added
product”. The intent of the vap variables is to provide engineering data
columns that are ready for analysis and visualization.

| DataFrame Variable | Long Name | Units |
|----|----|----|
| `u` | Eastward Sea Water Velocity | m/s |
| `v` | North Sea Water Velocity | m/s |
| `vap_sea_water_speed_layer_<0-9>` | Sea Water Speed at the specified sigma layer | m/s |
| `vap_sea_water_power_density_layer_<0-9>` | Sea Water Power Density (`rho = 1025`) | W/m2 |
| `vap_water_column_sea_water_speed` | Depth Averaged Sea Water Speed | m/s |
| `vap_sea_water_to_direction` | Sea Water To Direction | deg clockwise from true north |
| `vap_surface_elevation` | Surface Elevation relative to Mean Sea Level (MSL) | m |
| `vap_sigma_layer_depth` | Layer Depth relative to MSL | m |
| `vap_depth` | Sea Floor Depth relative to MSL | m |

<!--
&#10;```python
import pandas as pd
import matplotlib.pyplot as plt
import boto3
import re
from haversine import haversine, Unit
from botocore.exceptions import NoCredentialsError, ClientError
&#10;# Target coordinates
# Cook Inlet, Near Nikiski, AK
lat_point = 60.730452
lon_point = -151.443570
target_coords = (lat_point, lon_point)
&#10;# S3 configuration
s3_bucket = "us-tidal"
s3_prefix = "AK_cook_inlet/"
&#10;print(f"Target coordinates: {lat_point}°N, {lon_point}°W")
print(f"Searching S3 bucket: {s3_bucket}/{s3_prefix}")
&#10;def extract_coords_from_s3_key(s3_key):
    """Extract latitude and longitude from S3 key with lat= and lon= patterns"""
    &#10;    # Look for lat= and lon= patterns in the S3 key
    lat_match = re.search(r'lat=([+-]?\d+(?:\.\d+)?)', s3_key)
    lon_match = re.search(r'lon=([+-]?\d+(?:\.\d+)?)', s3_key)
    &#10;    if lat_match and lon_match:
        lat = float(lat_match.group(1))
        lon = float(lon_match.group(1))
        return lat, lon
    &#10;    return None, None
&#10;def find_closest_s3_file(bucket, prefix, target_coords):
    """Find the closest parquet file in S3 to target coordinates"""
    &#10;    try:
        # Initialize S3 client
        s3_client = boto3.client('s3')
        &#10;        # List all objects with the prefix
        paginator = s3_client.get_paginator('list_objects_v2')
        &#10;        closest_file = None
        min_distance = float('inf')
        file_coords = None
        parquet_files = []
        &#10;        print("Searching S3 for parquet files...")
        &#10;        # Paginate through all objects
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            if 'Contents' in page:
                for obj in page['Contents']:
                    key = obj['Key']
                    &#10;                    # Only process parquet files
                    if key.endswith('.parquet'):
                        parquet_files.append(key)
                        &#10;                        # Extract coordinates from filename
                        lat, lon = extract_coords_from_s3_key(key)
                        &#10;                        if lat is not None and lon is not None:
                            file_coord = (lat, lon)
                            # Calculate haversine distance
                            distance = haversine(target_coords, file_coord, unit=Unit.KILOMETERS)
                            &#10;                            print(f"File: {key.split('/')[-1]} | Coords: ({lat:.4f}, {lon:.4f}) | Distance: {distance:.2f} km")
                            &#10;                            if distance < min_distance:
                                min_distance = distance
                                closest_file = key
                                file_coords = file_coord
        &#10;        print(f"\nFound {len(parquet_files)} parquet files total")
        return closest_file, min_distance, file_coords
        &#10;    except NoCredentialsError:
        print("AWS credentials not found. Please configure your credentials.")
        return None, None, None
    except ClientError as e:
        print(f"Error accessing S3: {e}")
        return None, None, None
&#10;def load_s3_parquet(bucket, key):
    """Load parquet file from S3"""
    s3_path = f"s3://{bucket}/{key}"
    return pd.read_parquet(s3_path)
&#10;# Find closest file
closest_file, distance, coords = find_closest_s3_file(s3_bucket, s3_prefix, target_coords)
&#10;if closest_file:
    print(f"\nClosest file: {closest_file}")
    print(f"File coordinates: {coords[0]:.4f}°N, {coords[1]:.4f}°W")
    print(f"Distance from target: {distance:.2f} km")
    &#10;    try:
        # Load the closest file from S3
        print("\nLoading data from S3...")
        df = load_s3_parquet(s3_bucket, closest_file)
        print(f"Loaded data with shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        &#10;        # Check for zeta_center column (try different possible names)
        zeta_columns = [col for col in df.columns if 'zeta' in col.lower()]
        &#10;        if zeta_columns:
            zeta_col = zeta_columns[0]  # Use first zeta column found
            print(f"Using column: {zeta_col}")
            &#10;            # Plot the data
            plt.figure(figsize=(12, 6))
            df[zeta_col].plot()
            plt.title(f'Surface Elevation at ({coords[0]:.4f}°N, {coords[1]:.4f}°W)\n'
                     f'Distance from target: {distance:.2f} km')
            plt.xlabel('Time')
            plt.ylabel('Surface Elevation (m)')
            plt.grid(True)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
            &#10;            # Print statistics
            print(f"\n{zeta_col} Statistics:")
            print(df[zeta_col].describe())
            &#10;        else:
            print("No zeta column found. Available columns:")
            for col in df.columns:
                print(f"  - {col}")
                &#10;    except Exception as e:
        print(f"Error loading file from S3: {e}")
        print("Make sure you have proper AWS credentials configured.")
        &#10;else:
    print("No suitable files found with coordinate information in filenames")
```
&#10;-->

Additional examples will be provided including:

- Multi-location comparative analysis
- Depth-resolved velocity profiles
- Tidal energy resource assessment workflows
- Integration with marine energy converter models

### Promotion Strategy

**How NREL intends to promote this dataset**:

To be determined by NREL Marine Energy communications teams.

### Tags

**Relevant tags from
<https://github.com/awslabs/open-data-registry/blob/main/tags.yaml>**:

- analysis ready data
- coastal
- fluid-dynamics
- geospatial
- marine
- model
- netcdf
- ocean currents
- ocean velocity
- ocean sea surface height
- ocean simulation
- oceans
- parquet

Other relevant tags, not in list

- tidal energy
- marine energy
- sea water speed
- FVCOM
- resource assessment

# Citations

## Dataset Citation

\[3\] Z. Yang *et al.*, “High resolution tidal hindcast.” Marine and
Hydrokinetic Data Repository, National Renewable Energy Laboratory,
https://mhkdr.openei.org/submissions/632, 2025. Available:
<https://mhkdr.openei.org/submissions/632>

### BibTeX

    @misc{high_resolution_tidal_hindcast_2025,
    title = {{High Resolution Tidal Hindcast Dataset}},
    author = {Yang, Zhaoqing and Deb, Mithun and Wang, Taiping and Spicer, Preston and Simms, Andrew and Young, Ethan and Lawson, Mike},
    url = {https://mhkdr.openei.org/submissions/632},
    year = {2025},
    howpublished = {Marine and Hydrokinetic Data Repository, National Renewable Energy Laboratory, https://mhkdr.openei.org/submissions/632},
    }

## Location Specific References

The following publications provide validation studies, resource
characterization methodologies, and site-specific analyses for each
location in the dataset.

### Alaska, Aleutian Islands

\[4\] P. Spicer, Z. Yang, T. Wang, and M. Deb, “Spatially varying
seasonal modulation to tidal stream energy potential due to mixed tidal
regimes in the aleutian islands, AK,” *Renewable Energy*, p. 123564, May
2025, doi:
[10.1016/j.renene.2025.123564](https://doi.org/10.1016/j.renene.2025.123564).

### Alaska, Cook Inlet

\[5\] M. Deb, Z. Yang, and T. Wang, “Characterizing in-stream turbulent
flow for tidal energy converter siting in cook inlet, alaska,”
*Renewable Energy*, vol. 252, p. 123345, May 2025, doi:
[10.1016/j.renene.2025.123345](https://doi.org/10.1016/j.renene.2025.123345).

### Maine, Western Passage

\[6\] M. Deb, Z. Yang, T. Wang, and L. Kilcher, “Turbulence modeling to
aid tidal energy resource characterization in the western passage,
maine, USA,” *Renewable Energy*, vol. 219, Apr. 2023, doi:
[10.1016/j.renene.2023.04.100](https://doi.org/10.1016/j.renene.2023.04.100).

\[7\] Z. Yang *et al.*, “Modeling assessment of tidal energy extraction
in the western passage,” *Journal of Marine Science and Engineering*,
vol. 8, no. 6, 2020, doi:
[10.3390/jmse8060411](https://doi.org/10.3390/jmse8060411).

### New Hampshire, Piscataqua River

\[8\] P. Spicer, Z. Yang, T. Wang, and M. Deb, “Tidal energy extraction
modifies tidal asymmetry and transport in a shallow, well-mixed
estuary,” *Frontiers in Marine Science*, vol. 10, Sep. 2023, doi:
[10.3389/fmars.2023.1268348](https://doi.org/10.3389/fmars.2023.1268348).

### Washington, Puget Sound

\[9\] M. Deb, Z. Yang, K. Haas, and T. Wang, “Hydrokinetic tidal energy
resource assessment following international electrotechnical commission
guidelines,” *Renewable Energy*, vol. 229, p. 120767, Jun. 2024, doi:
[10.1016/j.renene.2024.120767](https://doi.org/10.1016/j.renene.2024.120767).

\[10\] P. Spicer, P. Maccready, and Z. Yang, “Localized tidal energy
extraction in puget sound can adjust estuary reflection and friction,
modifying barotropic tides system‐wide,” *Journal of Geophysical
Research: Oceans*, vol. 129, May 2024, doi:
[10.1029/2023JC020401](https://doi.org/10.1029/2023JC020401).

\[11\] Z. Yang, T. Wang, R. Branch, Z. Xiao, and M. Deb, “Tidal stream
energy resource characterization in the salish sea,” *Renewable Energy*,
vol. 172, Mar. 2021, doi:
[10.1016/j.renene.2021.03.028](https://doi.org/10.1016/j.renene.2021.03.028).

### Finite Volume Model Citation

\[1\] C. Chen, R. C. Beardsley, and G. Cowles, “An unstructured grid,
finite-volume coastal ocean model (FVCOM) system,” *Oceanography*, vol.
19, no. 1, pp. 78–89, 2006, doi:
[10.5670/oceanog.2006.92](https://doi.org/10.5670/oceanog.2006.92).

## All Citations and References

<div id="refs" class="references csl-bib-body" entry-spacing="0">

<div id="ref-fvcom" class="csl-entry">

<span class="csl-left-margin">\[1\]
</span><span class="csl-right-inline">C. Chen, R. C. Beardsley, and G.
Cowles, “An unstructured grid, finite-volume coastal ocean model (FVCOM)
system,” *Oceanography*, vol. 19, no. 1, pp. 78–89, 2006, doi:
[10.5670/oceanog.2006.92](https://doi.org/10.5670/oceanog.2006.92).</span>

</div>

<div id="ref-iec_62600_201" class="csl-entry">

<span class="csl-left-margin">\[2\]
</span><span class="csl-right-inline">International Electrotechnical
Commission, “<span class="nocase">Marine energy – Wave, tidal and other
water current converters – Part 201: Tidal energy resource assessment
and characterization</span>,” International Electrotechnical Commission,
Geneva, Switzerland, IEC/TS 62600-201 Ed. 1.0, Apr. 2015. Available:
<https://webstore.iec.ch/en/publication/22099></span>

</div>

<div id="ref-mhkdr_submission" class="csl-entry">

<span class="csl-left-margin">\[3\]
</span><span class="csl-right-inline">Z. Yang *et al.*, “High resolution
tidal hindcast.” Marine and Hydrokinetic Data Repository, National
Renewable Energy Laboratory, https://mhkdr.openei.org/submissions/632,
2025. Available: <https://mhkdr.openei.org/submissions/632></span>

</div>

<div id="ref-ak_aleutian_spicer2025_spatially" class="csl-entry">

<span class="csl-left-margin">\[4\]
</span><span class="csl-right-inline">P. Spicer, Z. Yang, T. Wang, and
M. Deb, “Spatially varying seasonal modulation to tidal stream energy
potential due to mixed tidal regimes in the aleutian islands, AK,”
*Renewable Energy*, p. 123564, May 2025, doi:
[10.1016/j.renene.2025.123564](https://doi.org/10.1016/j.renene.2025.123564).</span>

</div>

<div id="ref-ak_cook_deb2025_characterizing" class="csl-entry">

<span class="csl-left-margin">\[5\]
</span><span class="csl-right-inline">M. Deb, Z. Yang, and T. Wang,
“Characterizing in-stream turbulent flow for tidal energy converter
siting in cook inlet, alaska,” *Renewable Energy*, vol. 252, p. 123345,
May 2025, doi:
[10.1016/j.renene.2025.123345](https://doi.org/10.1016/j.renene.2025.123345).</span>

</div>

<div id="ref-ME_western_Deb2023_turbulence" class="csl-entry">

<span class="csl-left-margin">\[6\]
</span><span class="csl-right-inline">M. Deb, Z. Yang, T. Wang, and L.
Kilcher, “Turbulence modeling to aid tidal energy resource
characterization in the western passage, maine, USA,” *Renewable
Energy*, vol. 219, Apr. 2023, doi:
[10.1016/j.renene.2023.04.100](https://doi.org/10.1016/j.renene.2023.04.100).</span>

</div>

<div id="ref-ME_western_yang2020_modeling" class="csl-entry">

<span class="csl-left-margin">\[7\]
</span><span class="csl-right-inline">Z. Yang *et al.*, “Modeling
assessment of tidal energy extraction in the western passage,” *Journal
of Marine Science and Engineering*, vol. 8, no. 6, 2020, doi:
[10.3390/jmse8060411](https://doi.org/10.3390/jmse8060411).</span>

</div>

<div id="ref-nh_piscataqua_spicer2023_tidal" class="csl-entry">

<span class="csl-left-margin">\[8\]
</span><span class="csl-right-inline">P. Spicer, Z. Yang, T. Wang, and
M. Deb, “Tidal energy extraction modifies tidal asymmetry and transport
in a shallow, well-mixed estuary,” *Frontiers in Marine Science*, vol.
10, Sep. 2023, doi:
[10.3389/fmars.2023.1268348](https://doi.org/10.3389/fmars.2023.1268348).</span>

</div>

<div id="ref-wa_puget_deb2024_tidal_iec" class="csl-entry">

<span class="csl-left-margin">\[9\]
</span><span class="csl-right-inline">M. Deb, Z. Yang, K. Haas, and T.
Wang, “Hydrokinetic tidal energy resource assessment following
international electrotechnical commission guidelines,” *Renewable
Energy*, vol. 229, p. 120767, Jun. 2024, doi:
[10.1016/j.renene.2024.120767](https://doi.org/10.1016/j.renene.2024.120767).</span>

</div>

<div id="ref-wa_puget_spicer2024_localized" class="csl-entry">

<span class="csl-left-margin">\[10\]
</span><span class="csl-right-inline">P. Spicer, P. Maccready, and Z.
Yang, “Localized tidal energy extraction in puget sound can adjust
estuary reflection and friction, modifying barotropic tides
system‐wide,” *Journal of Geophysical Research: Oceans*, vol. 129, May
2024, doi:
[10.1029/2023JC020401](https://doi.org/10.1029/2023JC020401).</span>

</div>

<div id="ref-wa_puget_yang2021_tidal" class="csl-entry">

<span class="csl-left-margin">\[11\]
</span><span class="csl-right-inline">Z. Yang, T. Wang, R. Branch, Z.
Xiao, and M. Deb, “Tidal stream energy resource characterization in the
salish sea,” *Renewable Energy*, vol. 172, Mar. 2021, doi:
[10.1016/j.renene.2021.03.028](https://doi.org/10.1016/j.renene.2021.03.028).</span>

</div>

</div>
