config = {
    # Input and Output Directories
    "dir": {
        # Input Data Directories
        "input": {
            "original": "/projects/hindcastra/Tidal",
        },
        # Output Data Directories
        "output": {
            # Standardized data with qc
            "standardized": "/scratch/asimms/Tidal/a1_std",
            "vap": "/scratch/asimms/Tidal/b1_vap",
            "summary_vap": "/scratch/asimms/Tidal/b2_summary_vap",
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
            "siglay_center": "ocean_sigma/general_coordinate",
        },
    },
    "location_specification": {
        "aleutian_islands": {
            "output_name": "AK_aleutian_islands",
            "base_dir": "Aleutian_Islands_year",
            "files": ["*.nc"],
            "start_date": "2010-06-03 00:00:00",
            "end_date": "2011-06-02 23:59:59",
            "expected_delta_t_seconds": 3600,  # 60 min
            "coordinates": {"system": "latitude/longitude"},
        },
        "cook_inlet": {
            "output_name": "AK_cook_inlet",
            "base_dir": "Cook_Inlet_PNNL",
            "files": ["*.nc"],
            "start_date": "2005-01-01 00:00:00",
            "end_date": "2005-12-31 23:59:59",
            "expected_delta_t_seconds": 3600,  # 60 min
            "coordinates": {"system": "latitude/longitude"},
        },
        "piscataqua_river": {
            "output_name": "NH_piscataqua_river",
            "base_dir": "PIR_full_year",
            "files": ["*.nc"],
            "start_date": "2007-01-01 00:00:00",
            "end_date": "2007-12-31 23:59:59",
            "expected_delta_t_seconds": 1800,  # 30 min
            "coordinates": {"system": "utm", "zone": 19},
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
            "end_date": "2015-12-31 23:59:59",
            "expected_delta_t_seconds": 1800,  # 30 min
            "coordinates": {"system": "utm", "zone": 10},
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
            "end_date": "2017-12-31 23:59:59",
            "expected_delta_t_seconds": 1800,  # 30 min
            "coordinates": {"system": "utm", "zone": 10},
        },
    },
    "metadata": {
        "code_url": "https://github.com/nrel/marine_energy_resource_characterization/tidal/fvcom",
        "description": "",
        "conventions": "ME Data Pipeline Standards v1.0",
    },
    "derived_vap": {""},
}
