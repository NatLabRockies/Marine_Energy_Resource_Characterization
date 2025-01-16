config = {
    "dataset": {
        "name": "tidal_hindcast_fvcom",
        "version": "0.1.0",
        "issue_date": "2025-02-01",
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
        "lat": {
            "dtype": "float32",
            "coordinates": ["lon", "lat"],
            "dimensions": ["node"],
            "attributes": {
                "long_name": "nodal latitude",
                "standard_name": "latitude",
                "units": "degrees_north",
                "coverage_content_type": "coordinate",
                "valid_min": "-90",
                "valid_max": "90",
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
                "coverage_content_type": "coordinate",
                "valid_min": "-180",
                "valid_max": "180",
            },
        },
        # Face Center
        "latc": {
            "dtype": "float32",
            "coordinates": ["lonc", "latc"],
            "dimensions": ["face"],
            "attributes": {
                "long_name": "zonal latitude",
                "standard_name": "latitude",
                "units": "degrees_north",
                "coverage_content_type": "coordinate",
                "valid_min": "-90",
                "valid_max": "90",
            },
        },
        "lonc": {
            "dtype": "float32",
            "coordinates": ["lonc", "latc"],
            "dimensions": ["face"],
            "attributes": {
                "long_name": "zonal longitude",
                "standard_name": "longitude",
                "units": "degrees_east",
                "coverage_content_type": "coordinate",
                "valid_min": "-180",
                "valid_max": "180",
            },
        },
        "face": {
            "dtype": "int64",
            "coordinates": ["lonc", "latc"],
            "dimensions": ["face"],
            "attributes": {},
            "coverage_content_type": "coordinate",
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
            "dimensions": ["three", "face"],
            "attributes": {"long_name": "nodes surrounding element"},
            "coverage_content_type": "referenceInformation",
        },
        "three": {
            "dtype": "int64",
            "coordinates": [],
            "dimensions": ["three"],
            "attributes": {},
            "coverage_content_type": "referenceInformation",
        },
        "u": {
            "dtype": "float32",
            "coordinates": ["lonc", "latc", "time"],
            "dimensions": ["time", "siglay", "face"],
            "attributes": {
                "long_name": "Eastward Water Velocity",
                "standard_name": "eastward_sea_water_velocity",
                "units": "meters s-1",
                "grid": "fvcom_grid",
                "type": "data",
                "mesh": "fvcom_mesh",
                "location": "face",
                "coverage_content_type": "modelResult",
            },
        },
        "v": {
            "dtype": "float32",
            "coordinates": ["lonc", "latc", "time"],
            "dimensions": ["time", "siglay", "face"],
            "attributes": {
                "long_name": "Northward Water Velocity",
                "standard_name": "Northward_sea_water_velocity",
                "units": "meters s-1",
                "grid": "fvcom_grid",
                "type": "data",
                "mesh": "fvcom_mesh",
                "location": "face",
                "coverage_content_type": "modelResult",
            },
        },
    },
    "derived_vap_specification": {
        "speed": {
            "coordinates": ["lonc", "latc", "time"],
            "dimensions": ["time", "siglay", "face"],
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
        "from_direction": {
            "dtype": "float32",
            "coordinates": ["lonc", "latc", "time"],
            "dimensions": ["time", "siglay", "face"],
            "attributes": {
                "long_name": "Sea Water Velocity From Direction",
                "standard_name": "sea_water_velocity_from_direction",
                "units": "degree",
                "description": (
                    "A velocity is a vector quantity. "
                    'The phrase "from_direction" indicates the direction from which the '
                    "velocity vector is coming. The direction is a bearing in the usual "
                    "geographical sense, measured positive clockwise from due north."
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
            "coordinates": ["lonc", "latc", "time"],
            "dimensions": ["time", "siglay", "face"],
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
            "start_date": "2010-06-03 00:00:00",
            "end_date": "2011-06-02 23:00:00",
            # DatetimeIndex(['2011-06-07 00:00:00+00:00'], dtype='datetime64[ns, UTC]', freq=None)
            # 0   NaT
            "files_to_exclude": ["MD_AIS_west_hrBathy_0370.nc"],
            "expected_delta_t_seconds": 3600,  # 60 min
            "coordinates": {"system": "latitude/longitude"},
            "description": "",
            "partition_frequency": "M",  # Monthly, Roughly 73GB per file
        },
        "cook_inlet": {
            "output_name": "AK_cook_inlet",
            "base_dir": "Cook_Inlet_PNNL",
            "files": ["*.nc"],
            # Verifying cki_0366.nc...
            # DatetimeIndex(['2006-01-01 00:00:00+00:00'], dtype='datetime64[ns, UTC]', freq=None)
            # 0   NaT
            "files_to_exclude": ["cki_0366.nc"],
            "start_date": "2005-01-01 00:00:00",
            "end_date": "2005-12-31 23:00:00",
            "expected_delta_t_seconds": 3600,  # 60 min
            "coordinates": {"system": "latitude/longitude"},
            "description": "",
            "partition_frequency": "M",  # Monthly, Roughly 35GB per file
        },
        "piscataqua_river": {
            "output_name": "NH_piscataqua_river",
            "base_dir": "PIR_full_year",
            "files": ["*.nc"],
            # ValueError: Time verification failure in /kfs2/projects/hindcastra/Tidal/PIR_full_year/PIR_0368.nc. Delta t is different than 1800 seconds. Check timestamps below
            # DatetimeIndex(['2008-01-01 00:00:00+00:00'], dtype='datetime64[ns, UTC]', freq=None)
            "files_to_exclude": ["PIR_0368.nc"],
            "start_date": "2007-01-01 00:00:00",
            "end_date": "2007-12-31 23:30:00",
            "expected_delta_t_seconds": 1800,  # 30 min
            "coordinates": {"system": "utm", "zone": 19},
            "description": "",
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
            "start_date": "2015-01-01 00:00:00",
            # This dataset is missing one day!
            "end_date": "2015-12-30 23:30:00",
            "expected_delta_t_seconds": 1800,  # 30 min
            "coordinates": {"system": "utm", "zone": 10},
            "description": "",
            "partition_frequency": "D",  # Weekly
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
            "start_date": "2017-01-01 00:00:00",
            "end_date": "2017-12-31 23:30:00",
            "expected_delta_t_seconds": 1800,  # 30 min
            "coordinates": {"system": "utm", "zone": 10},
            "description": "",
            "partition_frequency": "M",  # Monthly, Roughly 50GB per file
        },
    },
    "metadata": {
        "code_url": "https://github.com/nrel/marine_energy_resource_characterization/tidal/fvcom",
        "cf_conventions": "1.0",
        "conventions": "ME Data Pipeline Standards v1.0",
        "data_generation_organization": "Pacific Northwest National Laboratory (PNNL)",
        "data_processing_organization": "National Renewable Energy Laboratory (NREL)",
        "funding_organization": "United States Department of Energy Water Power Technologies Office",
    },
}
