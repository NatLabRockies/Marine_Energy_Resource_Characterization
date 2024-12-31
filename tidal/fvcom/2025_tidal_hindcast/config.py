config = {
    # Input and Output Directories
    "dir": {
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
            "u": "eastward_sea_water_velocity",
            "v": "Northward_sea_water_velocity",  # The capital N is in the original data
            "zeta": "sea_surface_height_above_geoid",
            "h_center": "sea_floor_depth_below_geoid",
            "siglev_center": "ocean_sigma/general_coordinate",
        },
    },
    "derived_vap_specification": {""},
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
