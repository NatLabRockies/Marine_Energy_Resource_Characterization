config = {
    "dataset": {
        "name": "tidal_hindcast_fvcom",
        "version": "0.1.0",
        "issue_date": "2025-02-01",
        "encoding": {
            "time": {
                "units": "seconds since 1970-01-01",
                "calendar": "proleptic_gregorian",
                "dtype": "int64",
            }
        },
    },
    "code": {
        "version": "0.1.0",
    },
    "dir": {
        # Input and Output Directories
        # Input Data Directories
        "input": {
            "original": "/projects/hindcastra/Tidal",
        },
        # Output Data Directories
        "output": {
            "tracking": "/scratch/asimms/Tidal/<location>/z99_tracking",
            # Standardized data with qc
            "standardized": "/scratch/asimms/Tidal/<location>/a1_std",
            "standardized_partition": "/scratch/asimms/Tidal/<location>/a2_std_partition",
            "vap": "/scratch/asimms/Tidal/<location>/b1_vap",
            "summary_vap": "/scratch/asimms/Tidal/<location>/b2_summary_vap",
        },
    },
    "model_specification": {
        "model_version": "FVCOM_4.3.1",
        "conventions": "CF-1.0",
        "required_original_variables": {
            # Time
            "time": {
                "dtype": "float32",  # 53379.0
                "coordinates": ["time"],
                "dimensions": ["time"],
                "attributes": {
                    "long_name": "time",
                    "units": "days since 1858-11-17 00:00:00",
                    "format": "modified julian day (MJD)",
                    "time_zone": "UTC",
                },
            },
            "Times": {
                "dtype": "|S26",  # 2005-01-09T00:00:00.000000
                "coordinates": ["time"],
                "dimensions": ["time"],
                "attributes": {"time_zone": "UTC"},
            },
            # Nodal
            "lat": {
                "dtype": "float32",
                "coordinates": ["lon", "lat"],
                "dimensions": ["node"],
                "attributes": {
                    "long_name": "nodal latitude",
                    "standard_name": "latitude",
                    "units": "degrees_north",
                },
            },
            "lon": {
                "dtype": "float32",
                "coordinates": ["lon", "lat"],
                "dimensions": ["node"],
                "attributes": {
                    "long_name": "nodal longitude",
                    "standard_name": "longitude",
                    "units": "degrees_east",
                },
            },
            # Zonal Center
            "latc": {
                "dtype": "float32",
                "coordinates": ["lonc", "latc"],
                "dimensions": ["nele"],
                "attributes": {
                    "long_name": "zonal latitude",
                    "standard_name": "latitude",
                    "units": "degrees_north",
                },
            },
            "lonc": {
                "dtype": "float32",
                "coordinates": ["lonc", "latc"],
                "dimensions": ["nele"],
                "attributes": {
                    "long_name": "zonal longitude",
                    "standard_name": "longitude",
                    "units": "degrees_east",
                },
            },
            # Nodal
            "x": {
                "dtype": "float32",
                "coordinates": ["lon", "lat"],
                "dimensions": ["node"],
                "attributes": {"long_name": "nodal x-coordinate", "units": "meters"},
            },
            "y": {
                "dtype": "float32",
                "coordinates": ["lon", "lat"],
                "dimensions": ["node"],
                "attributes": {"long_name": "nodal y-coordinate", "units": "meters"},
            },
            # Zonal Center
            "xc": {
                "dtype": "float32",
                "coordinates": ["lonc", "latc"],
                "dimensions": ["nele"],
                "attributes": {"long_name": "zonal x-coordinate", "units": "meters"},
            },
            "yc": {
                "dtype": "float32",
                "coordinates": ["lonc", "latc"],
                "dimensions": ["nele"],
                "attributes": {"long_name": "zonal y-coordinate", "units": "meters"},
            },
            # Supporting Dimensions / Coordinates
            "nele": {
                "dtype": "int64",
                "coordinates": ["lonc", "latc"],
                "dimensions": ["nele"],
                "attributes": {},
            },
            "node": {
                "dtype": "int64",
                "coordinates": ["lon", "lat"],
                "dimensions": ["node"],
                "attributes": {},
            },
            "nv": {
                "dtype": "int32",
                "coordinates": ["lonc", "latc"],
                "dimensions": ["three", "nele"],
                "attributes": {"long_name": "nodes surrounding element"},
            },
            "three": {
                "dtype": "int64",
                "coordinates": [],
                "dimensions": ["three"],
                "attributes": {},
            },
            # Heights
            "zeta": {
                "dtype": "float32",
                "coordinates": ["lon", "lat", "time"],
                "dimensions": ["time", "node"],
                "attributes": {
                    "long_name": "Water Surface Elevation",
                    "units": "meters",
                    "positive": "up",
                    "standard_name": "sea_surface_height_above_geoid",
                    "grid": "Bathymetry_Mesh",
                    "type": "data",
                    "location": "node",
                },
            },
            "h_center": {
                "dtype": "float32",
                "coordinates": ["lonc", "latc"],
                "dimensions": ["nele"],
                "attributes": {
                    "long_name": "Bathymetry",
                    "standard_name": "sea_floor_depth_below_geoid",
                    "units": "m",
                    "positive": "down",
                    "grid": "grid1 grid3",
                    "grid_location": "center",
                },
            },
            # Depths
            "siglev_center": {
                "dtype": "float32",
                "coordinates": ["lonc", "latc"],
                "dimensions": ["siglev", "nele"],
                "attributes": {
                    "long_name": "Sigma Levels",
                    "standard_name": "ocean_sigma/general_coordinate",
                    "positive": "up",
                    "valid_min": -1.0,
                    "valid_max": 0.0,
                    "formula_terms": "sigma:siglay_center eta: zeta_center depth: h_center",
                },
            },
            # Model Output
            "u": {
                "dtype": "float32",
                "coordinates": ["lonc", "latc", "time"],
                "dimensions": ["time", "siglay", "nele"],
                "attributes": {
                    "long_name": "Eastward Water Velocity",
                    "standard_name": "eastward_sea_water_velocity",
                    "units": "meters s-1",
                    "grid": "fvcom_grid",
                    "type": "data",
                    "mesh": "fvcom_mesh",
                    "location": "face",
                },
            },
            "v": {
                "dtype": "float32",
                "coordinates": ["lonc", "latc", "time"],
                "dimensions": ["time", "siglay", "nele"],
                "attributes": {
                    "long_name": "Northward Water Velocity",
                    "standard_name": "Northward_sea_water_velocity",
                    "units": "meters s-1",
                    "grid": "fvcom_grid",
                    "type": "data",
                    "mesh": "fvcom_mesh",
                    "location": "face",
                },
            },
        },
    },
    "standardized_variable_specification": {
        "time": {
            "dtype": "datetime64[ns]",
            "coordinates": ["time"],
            "dimensions": ["time"],
            "attributes": {
                "standard_name": "time",
                "long_name": "Time",
                "time_zone": "UTC",
                "coverage_content_type": "coordinate",
            },
        },
        "lat_node": {
            "dtype": "float32",
            "coordinates": ["lat_node", "lon_node"],
            "dimensions": ["node"],
            "attributes": {
                "long_name": "Nodal Latitude",
                "standard_name": "latitude",
                "units": "degrees_north",
                "coverage_content_type": "coordinate",
                "valid_min": "-90",
                "valid_max": "90",
            },
        },
        "lon_node": {
            "dtype": "float32",
            "coordinates": ["lon_node", "lat_node"],
            "dimensions": ["node"],
            "attributes": {
                "long_name": "Nodal Longitude",
                "standard_name": "longitude",
                "units": "degrees_east",
                "coverage_content_type": "coordinate",
                "valid_min": "-180",
                "valid_max": "180",
            },
        },
        # Face Center
        "lat_center": {
            "dtype": "float32",
            "coordinates": ["lat_center", "lon_center"],
            "dimensions": ["face"],
            "attributes": {
                "long_name": "Face Center Latitude",
                "standard_name": "latitude",
                "units": "degrees_north",
                "coverage_content_type": "coordinate",
                "valid_min": "-90",
                "valid_max": "90",
            },
        },
        "lon_center": {
            "dtype": "float32",
            "coordinates": ["lat_center", "lon_center"],
            "dimensions": ["face"],
            "attributes": {
                "long_name": "Face Center Longitude",
                "standard_name": "longitude",
                "units": "degrees_east",
                "coverage_content_type": "coordinate",
                "valid_min": "-180",
                "valid_max": "180",
            },
        },
        "face": {
            "dtype": "int64",
            "coordinates": ["lat_center", "lon_center"],
            "dimensions": ["face"],
            "attributes": {},
            "coverage_content_type": "coordinate",
        },
        "node": {
            "dtype": "int64",
            "coordinates": ["lat_node", "lon_node"],
            "dimensions": ["node"],
            "attributes": {},
        },
        "nv": {
            "dtype": "int32",
            "coordinates": ["lat_center", "lon_center"],
            "dimensions": ["face_node_index", "face"],
            "attributes": {"long_name": "nodes surrounding element"},
            "coverage_content_type": "referenceInformation",
        },
        "face_node": {
            "dtype": "int64",
            "coordinates": ["face_node_index"],
            "dimensions": ["face_node"],
            "attributes": {"long_name": "nodes surrounding element"},
            "coverage_content_type": "referenceInformation",
        },
        "face_node_index": {
            "dtype": "int64",
            "coordinates": [],
            "dimensions": ["face_node"],
            "attributes": {},
            "coverage_content_type": "referenceInformation",
        },
        "u": {
            "dtype": "float32",
            "coordinates": ["lat_center", "lon_center", "time"],
            "dimensions": ["time", "sigma_layer", "face"],
            "attributes": {
                "long_name": "Eastward Water Velocity",
                "standard_name": "eastward_sea_water_velocity",
                "units": "m s-1",
                "grid": "fvcom_grid",
                "type": "data",
                "mesh": "fvcom_mesh",
                "location": "face",
                "coverage_content_type": "modelResult",
            },
        },
        "v": {
            "dtype": "float32",
            "coordinates": ["lat_center", "lon_center", "time"],
            "dimensions": ["time", "sigma_layer", "face"],
            "attributes": {
                "long_name": "Northward Water Velocity",
                "standard_name": "northward_sea_water_velocity",
                "units": "m s-1",
                "grid": "fvcom_grid",
                "type": "data",
                "mesh": "fvcom_mesh",
                "location": "face",
                "coverage_content_type": "modelResult",
            },
        },
        "zeta": {
            "dtype": "float32",
            "coordinates": ["lat_node", "lon_node", "time"],
            "dimensions": ["time", "node"],
            "attributes": {
                "long_name": "Water Surface Elevation",
                "standard_name": "sea_surface_height_above_geoid",
                "units": "m",
                "positive": "up",
                "grid": "Bathymetry_Mesh",
                "type": "data",
                "location": "face",
                "coverage_content_type": "modelResult",
            },
        },
        "h_center": {
            "dtype": "float32",
            "coordinates": ["lat_center", "lon_center"],
            "dimensions": ["face"],
            "attributes": {
                "long_name": "Bathymetry",
                "standard_name": "sea_surface_depth_below_geoid",
                "units": "m",
                "positive": "down",
                "grid": "grid1 grid3",
                "grid_location": "center",
                "type": "data",
                "location": "face",
                "coverage_content_type": "modelResult",
            },
        },
    },
    "derived_vap_specification": {
        "speed": {
            "coordinates": ["lat_center", "lon_center", "time"],
            "dimensions": ["time", "sigma_layer", "face"],
            "attributes": {
                "long_name": "Sea Water Speed",
                "standard_name": "sea_water_speed",
                "units": "m s-1",
                "description": "Speed is the magnitude of velocity.",
                "grid": "fvcom_grid",
                "type": "data",
                "mesh": "fvcom_mesh",
                "location": "face",
                "coverage_content_type": "modelResult",
            },
        },
        "to_direction": {
            "dtype": "float32",
            "coordinates": ["lat_center", "lon_center", "time"],
            "dimensions": ["time", "sigma_layer", "face"],
            "attributes": {
                "long_name": "Sea Water Velocity To Direction ",
                "standard_name": "sea_water_velocity_to_direction",
                "units": "degree",
                "description": (
                    "A velocity is a vector quantity. "
                    "The phrase 'to_direction' indicates the direction toward which the "
                    "velocity vector is pointing (the destination of flow). The direction is a bearing in the usual "
                    "geographical sense, measured positive clockwise from true north."
                ),
                "valid_min": "0.0",
                "valid_max": "360.0",
                "grid": "fvcom_grid",
                "type": "data",
                "mesh": "fvcom_mesh",
                "location": "face",
                "coverage_content_type": "modelResult",
            },
        },
        "from_direction": {
            "dtype": "float32",
            "coordinates": ["lat_center", "lon_center", "time"],
            "dimensions": ["time", "sigma_layer", "face"],
            "attributes": {
                "long_name": "Sea Water Velocity From Direction",
                "standard_name": "sea_water_velocity_from_direction",
                "units": "degree",
                "description": (
                    "A velocity is a vector quantity. "
                    "The phrase 'from_direction' indicates the direction from which the "
                    "velocity vector is coming (the source of flow). The direction is a bearing in the usual "
                    "geographical sense, measured positive clockwise from true north."
                ),
                "valid_min": "0.0",
                "valid_max": "360.0",
                "grid": "fvcom_grid",
                "type": "data",
                "mesh": "fvcom_mesh",
                "location": "face",
                "coverage_content_type": "modelResult",
            },
        },
        "power_density": {
            "dtype": "float32",
            "coordinates": ["lat_center", "lon_center", "time"],
            "dimensions": ["time", "sigma_layer", "face"],
            "attributes": {
                "long_name": "Sea Water Power Density",
                "units": "W m-2",
                "grid": "fvcom_grid",
                "type": "data",
                "mesh": "fvcom_mesh",
                "location": "face",
                "coverage_content_type": "modelResult",
            },
        },
    },
    "time_specification": {
        # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.drop_duplicates.html
        "drop_duplicate_timestamps_keep_strategy": "first",
    },
    "location_specification": {
        "aleutian_islands": {
            "output_name": "AK_aleutian_islands",
            "base_dir": "Aleutian_Islands_year",
            "files": ["*.nc"],
            "start_date_utc": "2010-06-03 00:00:00",
            "end_date_utc": "2011-06-02 23:00:00",
            # DatetimeIndex(['2011-06-07 00:00:00+00:00'], dtype='datetime64[ns, UTC]', freq=None)
            # 0   NaT
            "files_to_exclude": ["MD_AIS_west_hrBathy_0370.nc"],
            "expected_delta_t_seconds": 3600,  # 60 min
            "temporal_resolution": "hourly",
            "coordinates": {"system": "latitude/longitude"},
            "description": "",
            "summary": "",
            # "partition_frequency": "M",  # Monthly, Roughly 73GB per file, out of memory in partition step
            "partition_frequency": "W",  # Weekly, Roughly ? per file
        },
        "cook_inlet": {
            "output_name": "AK_cook_inlet",
            "base_dir": "Cook_Inlet_PNNL",
            "files": ["*.nc"],
            # Verifying cki_0366.nc...
            # DatetimeIndex(['2006-01-01 00:00:00+00:00'], dtype='datetime64[ns, UTC]', freq=None)
            # 0   NaT
            "files_to_exclude": ["cki_0366.nc"],
            "start_date_utc": "2005-01-01 00:00:00",
            "end_date_utc": "2005-12-31 23:00:00",
            "expected_delta_t_seconds": 3600,  # 60 min
            "temporal_resolution": "hourly",
            "coordinates": {"system": "latitude/longitude"},
            "description": "",
            "summary": "",
            "partition_frequency": "M",  # Monthly, Roughly 35GB per file
        },
        "piscataqua_river": {
            "output_name": "NH_piscataqua_river",
            "base_dir": "PIR_full_year",
            "files": ["*.nc"],
            # ValueError: Time verification failure in /kfs2/projects/hindcastra/Tidal/PIR_full_year/PIR_0368.nc. Delta t is different than 1800 seconds. Check timestamps below
            # DatetimeIndex(['2008-01-01 00:00:00+00:00'], dtype='datetime64[ns, UTC]', freq=None)
            "files_to_exclude": ["PIR_0368.nc"],
            "start_date_utc": "2007-01-01 00:00:00",
            "end_date_utc": "2007-12-31 23:30:00",
            "expected_delta_t_seconds": 1800,  # 30 min
            "temporal_resolution": "half-hourly",
            "coordinates": {"system": "utm", "zone": 19},
            "description": "",
            "summary": "",
            "partition_frequency": "M",  # Monthly, Roughly 67GB per file
        },
        "puget_sound": {
            "output_name": "WA_puget_sound",
            "base_dir": "Puget_Sound_corrected",
            "files": [
                "02012015/*.nc",
                "03012015/*.nc",
                "04012015/*.nc",
                "05012015/*.nc",
                "06012015/*.nc",
                "07012015/*.nc",
                "08012015/*.nc",
                "09012015/*.nc",
                "10012015/*.nc",
                "11012015/*.nc",
                "12012015/*.nc",
                "12312015/*.nc",
            ],
            "start_date_utc": "2015-01-01 00:00:00",
            # This dataset is missing one day!
            "end_date_utc": "2015-12-30 23:30:00",
            "expected_delta_t_seconds": 1800,  # 30 min
            "temporal_resolution": "half-hourly",
            "coordinates": {"system": "utm", "zone": 10},
            "description": "",
            "summary": "",
            # "partition_frequency": "D",  # Weekly, out of memory at vap step
            "partition_frequency": "D",  # Daily
        },
        "western_passage": {
            "output_name": "ME_western_passage",
            "base_dir": "Western_Passage_corrected",
            "files": [
                "01_Jan_Mar/*.nc",
                "02_Apr_Jun/*.nc",
                "03_Jul_Sep/*.nc",
                "04_Oct_Dec/*.nc",
            ],
            "start_date_utc": "2017-01-01 00:00:00",
            "end_date_utc": "2017-12-31 23:30:00",
            "expected_delta_t_seconds": 1800,  # 30 min
            "temporal_resolution": "half-hourly",
            "coordinates": {"system": "utm", "zone": 19},
            "title": "Western Passage High Fidelity Tidal Hindcast",
            "description": "",
            "summary": "",
            # "partition_frequency": "M",  # Monthly, Roughly 50GB per file
            # Trying to improve VAP compute time and find a balance between # of files and file size
            "partition_frequency": "12h",
        },
    },
    "global_attributes": {
        # Global attributes that will be recorded in the output dataset in `attrs` (attributes). These metadata are
        # used to record data provenance information (e.g., location, institution, etc),
        # construct datastream and file names (i.e., location_id, dataset_name, qualifier,
        # temporal, and data_level attributes), as well as provide metadata that is useful for
        # data users (e.g., title, description, ... ).
        # Source: ACDD
        # A comma-separated list of the conventions that are followed by the dataset.
        # For files that follow this version of ACDD, include the string 'ACDD-1.3'.
        "Conventions": "CF-1.0, ACDD-1.3, ME Data Pipeline-1.0",
        # Source: ACDD
        # A place to acknowledge various types of support for the project that produced this data.
        "acknowledgement": "This work was funded by the U.S. Department of Energy, Office of "
        "Energy Efficiency & Renewable Energy, Water Power Technologies Office. The authors "
        "gratefully acknowledge project support from Heather Spence and Jim McNally (U.S. "
        "Department of Energy Water Power Technologies Office) and Mary Serafin (National "
        "Renewable Energy Laboratory). Technical guidance was provided by Vincent Neary (Sandia "
        "National Laboratories), Levi Kilcher, Caroline Draxl, and Katie Peterson (National "
        "Renewable Energy Laboratory).",
        # Source: IOOS
        # Country of the person or organization that operates a platform or network,
        # which collected the observation data.
        "creator_country": "USA",
        # Source: ACDD, IOOS
        # The email address of the person (or other creator type specified by the creator_type
        # attribute) principally responsible for creating this data.
        "creator_email": "zhaoqing.yang@pnnl.gov",
        # Source: ACDD, IOOS
        # The institution of the creator; should uniquely identify the creator's institution.
        # This attribute's value should be specified even if it matches the value of
        # publisher_institution, or if creator_type is institution.
        "creator_institution": "Pacific Northwest National Laboratory (PNNL)",
        # Source: IOOS
        # URL for the institution that collected the data. For clarity, it is recommended
        # that this field is specified even if the creator_type is institution and a
        # creator_url is provided.
        "creator_institution_url": "https://www.pnnl.gov/",
        # Source: ACDD
        # The name of the person (or other creator type specified by the creator_type
        # attribute) principally responsible for creating this data.
        "creator_name": "Zhaoqing Yang",
        # Source: IOOS
        # IOOS classifier (https://mmisw.org/ont/ioos/sector) that best describes
        # the platform (network) operator's societal sector.
        "creator_sector": "gov_federal",
        # Source: IOOS
        # State or province of the person or organization that collected the data.
        "creator_state": "Washington",
        # Source: ACDD
        # Specifies type of creator with one of the following: 'person', 'group',
        # 'institution', or 'position'. If this attribute is not specified, the
        # creator is assumed to be a person.
        "creator_type": "institution",
        # Source: ACDD, IOOS
        # The URL of the person (or other creator type specified by the creator_type
        # attribute) principally responsible for creating this data.
        "creator_url": "https://www.pnnl.gov/",
        # Source: ACDD
        # The name of any individuals, projects, or institutions that contributed to
        # the creation of this data.
        "contributor_name": "Mithun Deb, Andrew Simms, Ethan Young",
        # Source: ACDD
        # The role of any individuals, projects, or institutions that contributed to
        # the creation of this data.
        # Contributor Role Definitions: "https://vocab.nerc.ac.uk/collection/G04/current/".
        # | Role                    | Definition                                                                                                       |
        # | ----------------------- | ---------------------------------------------------------------------------------------------------------------- |
        # | author                  | Party who authored the resource                                                                                  |
        # | coAuthor                | Party who jointly authors the resource                                                                           |
        # | collaborator            | Party who assists with the generation of the resource other than the principal investigator                      |
        # | contributor             | Party contributing to the resource                                                                               |
        # | custodian               | Party that accepts accountability and responsibility for the data, and ensures appropriate care and maintenance  |
        # | distributor             | Party who distributes the resource                                                                               |
        # | editor                  | Party who reviewed or modified the resource to improve the content                                               |
        # | funder                  | Party providing monetary support for the resource                                                                |
        # | mediator                | A class of entity that mediates access to the resource and for whom the resource is intended or useful           |
        # | originator              | Party who created the resource                                                                                   |
        # | owner                   | Party that owns the resource                                                                                     |
        # | pointOfContact          | Party who can be contacted for acquiring knowledge about or acquisition of the resource                          |
        # | principalInvestigator   | Key party responsible for gathering information and conducting research                                          |
        # | processor               | Party that has processed the data in a manner such that the resource has been modified                           |
        # | publisher               | Party who published the resource                                                                                 |
        # | resourceProvider        | Party that supplies the resource                                                                                 |
        # | rightsHolder            | Party owning or managing rights over the resource                                                                |
        # | sponsor                 | Party who speaks for the resource                                                                                |
        # | stakeholder             | Party who has an interest in the resource or the use of the resource                                             |
        # | user                    | Party who uses the resource                                                                                      |
        "contributor_role": "author, processor, processor",
        # Source: IOOS
        # The URL of the controlled vocabulary used for the contributor_role attribute.
        # The default is "https://vocab.nerc.ac.uk/collection/G04/current/".
        "contributor_role_vocabulary": "https://vocab.nerc.ac.uk/collection/G04/current/",
        # Source: IOOS
        # The URL of the individuals or institutions that contributed to the creation
        # of this data.
        "contributor_url": "https://www.pnnl.gov, www.nrel.gov",
        # Source: Global, ACDD
        # A user-friendly description of the dataset. It should provide enough context
        # about the data for new users to quickly understand how the data can be used.
        "description": None,
        # Source: Global
        # The DOI that has been registered for this dataset, if applicable.
        "doi": None,
        # Source: Global, ACDD, IOOS
        # CF attribute for identifying the featureType.
        # https://cfconventions.org/Data/cf-conventions/cf-conventions-1.12/cf-conventions.html#table-feature-types
        "featureType": "timeSeries",
        # Source: IOOS
        # URL for background information about this dataset.
        "infoURL": "https://www.github.com/nrel/marine_energy_resource_characterization/tidal/fvcom/fy_25_tidal_hindcast",
        # Source: ACDD
        # Name of the contributing instrument(s) or sensor(s) used to create this
        # data set or product.
        "instrument": None,
        # Source: ACDD
        # Controlled vocabulary for the names used in the 'instrument' attribute.
        "instrument_vocabulary": None,
        # Source: ACDD
        # A comma-separated list of key words and/or phrases.
        # "keywords": "OCEAN WAVES, GRAVITY WAVES, WIND WAVES, SIGNIFICANT WAVE HEIGHT, WAVE FREQUENCY, WAVE PERIOD, WAVE SPECTRA,",
        "keywords": "OCEAN TIDES, TIDAL ENERGY, VELOCITY, SPEED, DIRECTION, POWER DENSITY",
        # Source: ACDD
        # If you are using a controlled vocabulary for the words/phrases in your
        # 'keywords' attribute, this is the unique name or identifier of the
        # vocabulary from which keywords are taken.
        # Excluded by PNNL
        "keywords_vocabulary": None,
        # Source: ACDD, IOOS
        # Provide the URL to a standard or specific license, enter 'Freely
        # Distributed' or 'None', or describe any restrictions to data access
        # and distribution in free text.
        "license": "Freely Distributed",
        # Source: Global
        # A label or acronym for the location where the data were obtained from.
        # Only alphanumeric characters and '_' are allowed.
        # "location_id": "cpr",
        # "make_model": "Sofar Spotter2",
        # Source: ACDD, IOOS
        # The organization that provides the initial id for the dataset.
        # "naming_authority": "gov.pnnl.sequim",
        "naming_authority": "gov.nrel.water_power",
        # Source: ACDD, IOOS
        # Name of the platform(s) that supported the sensor data used to create this
        # data set or product.
        # "platform": "wave_buoy",
        # Source: IOOS
        # An optional, short identifier for the platform, if the data provider
        # prefers to define an id that differs from the dataset identifier.
        # "platform_id": None,
        # Source: IOOS
        # A descriptive, long name for the platform used in collecting the data.
        # "platform_name": None,
        # Source: ACDD, IOOS
        # Controlled vocabulary for the names used in the 'platform' attribute.
        # "platform_vocabulary": "http://mmisw.org/ont/ioos/platform",
        # Source: Global
        # Optional attribute used to cite other data, algorithms, etc. as needed.
        "references": "Spicer, P., Z. Yang, T. Wang, and M. Deb. (in prep.)."
        "Considering the relative importance of diurnal and semidiurnal tides "
        "in tidal power potential around the Aleutian Islands, AK, Renewable Energy.",
        # Source: Global
        # An optional string which distinguishes these data from other datasets
        # produced by the same instrument.
        # "qualifier": None,
        # Source: ACDD
        # The data type, as derived from Unidata's Common Data Model Scientific Data types.
        # "cdm_data_type": None,
        # Source: ACDD
        # Miscellaneous information about the data, not captured elsewhere.
        "comment": None,
        # Source: ACDD
        # A URL that gives the location of more complete metadata.
        "metadata_link": None,
        # Source: ACDD
        # The overarching program(s) of which the dataset is a part.
        "program": "DOE Water Power Technologies Office Marine Energy Resource Characterization",
        # Source: ACDD
        # The name of the project(s) principally responsible for originating
        # this data.
        "project": "High Resolution Tidal Hindcast",
        # Source: ACDD
        # A paragraph describing the dataset, analogous to an abstract for a paper.
        "summary": None,
        # Source: IOOS
        # Country of the person or organization that distributes the data.
        "publisher_country": "USA",
        # Source: ACDD, IOOS
        # The email address of the person responsible for publishing the data file
        # or product to users.
        "publisher_email": "michael.lawson@nrel.gov",
        # Source: ACDD, IOOS
        # The institution of the publisher.
        "publisher_institution": "Pacific Northwest National Laboratory (PNNL)",
        # Source: ACDD
        # The name of the person responsible for publishing the data file or
        # product to users.
        "publisher_name": "Michael Lawson",
        # Source: IOOS
        # State or province of the person or organization that distributes the data.
        "publisher_state": "Colorado",
        # Source: ACDD
        # Specifies type of publisher with one of the following: 'person', 'group',
        # 'institution', or 'position'.
        "publisher_type": "institution",
        # Source: ACDD, IOOS
        # The URL of the person responsible for publishing the data file or product
        # to users.
        "publisher_url": "https://www.nrel.gov",
    },
}
