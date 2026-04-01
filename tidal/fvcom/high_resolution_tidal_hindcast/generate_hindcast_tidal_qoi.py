import argparse
import glob
from datetime import datetime

import numpy as np
import pandas as pd
import utm
import xarray as xr

DATA_DIR = "/projects/hindcastra/Tidal"
OUTPUT_DIR = "/scratch/asimms"

#  Output Path Manager --------------------------------------------------{{{


def generate_output_nc_path(set_name, details=None):
    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
    if details is not None:
        details = f".{details}"

    return f"{OUTPUT_DIR}/{set_name}{details}.{formatted_datetime}.nc"


#  End Output Path Manager ----------------------------------------------}}}
#  Individual Dataset Specifications -----------------------------------{{{


def generate_western_passage_corrected_spec():
    base_dir = f"{DATA_DIR}/Western_Passage_corrected"
    path_to_files = []
    path_to_files.extend(sorted(glob.glob(f"{base_dir}/01_Jan_Mar/*.nc")))
    path_to_files.extend(sorted(glob.glob(f"{base_dir}/02_Apr_Jun/*.nc")))
    path_to_files.extend(sorted(glob.glob(f"{base_dir}/03_Jul_Sep/*.nc")))
    path_to_files.extend(sorted(glob.glob(f"{base_dir}/04_Oct_Dec/*.nc")))

    # .to_numpy() extracts  np datetime64[ns] values from datetime conversion,
    # necessary for comparing and filtering by these dates
    start_date = pd.to_datetime("2017-01-01 00:00:00", utc=True).to_numpy()
    end_date = pd.to_datetime("2017-12-31 23:59:59", utc=True).to_numpy()

    western_passage_utm_zone = 19
    lat, lon = convert_utm_coordinates(path_to_files, western_passage_utm_zone, True)

    return {
        "input_files": path_to_files,
        "filter_start_date": start_date,
        "filter_end_date": end_date,
        "lat": lat,
        "lon": lon,
        "output_filepath": generate_output_nc_path(
            "Western_Passage_corrected",
            "2017_all_time-half_hourly-current_speed-tidal_power_density",
        ),
    }


def generate_puget_sound_corrected_spec():
    base_dir = f"{DATA_DIR}/Puget_Sound_corrected"
    path_to_files = []
    # 02012015  03012015  04012015  05012015  06012015  07012015  08012015  09012015  10012015  11012015  12012015  12312015
    path_to_files.extend(sorted(glob.glob(f"{base_dir}/02012015/*.nc")))
    path_to_files.extend(sorted(glob.glob(f"{base_dir}/03012015/*.nc")))
    path_to_files.extend(sorted(glob.glob(f"{base_dir}/04012015/*.nc")))
    path_to_files.extend(sorted(glob.glob(f"{base_dir}/05012015/*.nc")))
    path_to_files.extend(sorted(glob.glob(f"{base_dir}/06012015/*.nc")))
    path_to_files.extend(sorted(glob.glob(f"{base_dir}/07012015/*.nc")))
    path_to_files.extend(sorted(glob.glob(f"{base_dir}/08012015/*.nc")))
    path_to_files.extend(sorted(glob.glob(f"{base_dir}/09012015/*.nc")))
    path_to_files.extend(sorted(glob.glob(f"{base_dir}/10012015/*.nc")))
    path_to_files.extend(sorted(glob.glob(f"{base_dir}/11012015/*.nc")))
    path_to_files.extend(sorted(glob.glob(f"{base_dir}/12012015/*.nc")))
    path_to_files.extend(sorted(glob.glob(f"{base_dir}/12312015/*.nc")))

    # .to_numpy() extracts  np datetime64[ns] values from datetime conversion,
    # necessary for comparing and filtering by these dates
    start_date = pd.to_datetime("2015-01-01 00:00:00", utc=True).to_numpy()
    end_date = pd.to_datetime("2015-12-31 23:59:59", utc=True).to_numpy()

    puget_sound_utm_zone = 10  # Seattle Area
    lat, lon = convert_utm_coordinates(path_to_files, puget_sound_utm_zone, True)

    return {
        "input_files": path_to_files,
        "filter_start_date": start_date,
        "filter_end_date": end_date,
        "lat": lat,
        "lon": lon,
        "output_filepath": generate_output_nc_path(
            "Puget_Sound_corrected",
            "2015_all_time-half_hourly-current_speed-tidal_power_density",
        ),
    }


def generate_cook_inlet_spec():
    base_dir = f"{DATA_DIR}/Cook_Inlet_PNNL"
    path_to_files = []
    path_to_files.extend(sorted(glob.glob(f"{base_dir}/*.nc")))

    # .to_numpy() extracts  np datetime64[ns] values from datetime conversion,
    # necessary for comparing and filtering by these dates
    start_date = pd.to_datetime("2005-01-01 00:00:00", utc=True).to_numpy()
    end_date = pd.to_datetime("2005-12-31 23:59:59", utc=True).to_numpy()

    return {
        "input_files": path_to_files,
        "filter_start_date": start_date,
        "filter_end_date": end_date,
        "output_filepath": generate_output_nc_path(
            "Cook_Inlet_PNNL",
            "2005_all_time-hourly-current_speed-tidal_power_density",
        ),
    }


def generate_pir_spec():
    base_dir = f"{DATA_DIR}/PIR_full_year"
    path_to_files = []
    path_to_files.extend(sorted(glob.glob(f"{base_dir}/*.nc")))

    # .to_numpy() extracts  np datetime64[ns] values from datetime conversion,
    # necessary for comparing and filtering by these dates
    start_date = pd.to_datetime("2007-01-01 00:00:00", utc=True).to_numpy()
    end_date = pd.to_datetime("2007-12-31 23:59:59", utc=True).to_numpy()

    pir_utm_zone = 19
    lat, lon = convert_utm_coordinates(path_to_files, pir_utm_zone, True)

    return {
        "input_files": path_to_files,
        "filter_start_date": start_date,
        "filter_end_date": end_date,
        "lat": lat,
        "lon": lon,
        "output_filepath": generate_output_nc_path(
            "PIR_full_year",
            "2007_all_time-half_hourly-current_speed-tidal_power_density",
        ),
    }


def generate_aleutian_spec():
    base_dir = f"{DATA_DIR}/Aleutian_Islands_year"
    path_to_files = []
    path_to_files.extend(sorted(glob.glob(f"{base_dir}/*.nc")))

    # .to_numpy() extracts  np datetime64[ns] values from datetime conversion,
    # necessary for comparing and filtering by these dates
    start_date = pd.to_datetime("2010-06-03 00:00:00", utc=True).to_numpy()
    end_date = pd.to_datetime("2011-06-02 23:59:59", utc=True).to_numpy()

    return {
        "input_files": path_to_files,
        "filter_start_date": start_date,
        "filter_end_date": end_date,
        "output_filepath": generate_output_nc_path(
            "Aleutian_Islands_year",
            "2010_2011_full_year-hourly-current_speed-tidal_power_density",
        ),
    }


#  End Individual Dataset Specifications -------------------------------}}}


def convert_utm_coordinates(path_to_files, utm_zone: int, utm_northern):
    first_ds = xr.open_dataset(path_to_files[0], decode_times=False)

    # This should be the same for all datasets
    this_lat = first_ds["latc"]
    this_lon = first_ds["lonc"]

    if not np.allclose(this_lat, 0.0) or not np.allclose(this_lon, 0.0):
        raise ValueError(
            "Aborting conversion from UTM to lat/lon. Original lat/lon is not 0.0"
        )

    xc = first_ds["xc"]
    yc = first_ds["yc"]

    return utm.to_latlon(xc, yc, utm_zone, northern=utm_northern)


# def cleanup_unwrapped_lat_lon(path_to_files):
#     first_ds = xr.open_dataset(path_to_files[0], decode_times=False)

#     this_lat = first_ds["latc"]
#     this_lon = first_ds["lonc"]

#     # Convert longitudes above 180 to -180 and start counting backwards
#     # e.g., 200 gets converted to -160
#     unwrapped_lon = np.where(
#         this_lon.values > 180, this_lon.values - 360, this_lon.values
#     )
#     this_lon.values = unwrapped_lon

#     return this_lat, this_lon


# def standardize_lat_lon(ds):
#     # Per ME Data Pipeline Standards: Version 1.0
#     # Section 4.3.4
#
#     ds = ds.rename({"latc": "latitude", "lonc": "longitude"})


def validate_and_standardize_time(
    ds, expected_delta_t_seconds, non_standard_time_variable_name=None
):
    """
    Validates and standardizes time coordinates according to CF conventions.

    Parameters:
    ds (xarray.Dataset): Input dataset

    Returns:
    xarray.Dataset: Dataset with standardized time coordinate
    """

    time_accessor = "time"

    if non_standard_time_variable_name is not None:
        time_accessor = non_standard_time_variable_name

        # Convert string times to datetime64
    times_string = [ts.decode("utf-8") for ts in ds[time_accessor].values]
    utc_datetimes = pd.to_datetime(times_string, utc=True).values

    # Create new time coordinate with CF attributes
    new_time = xr.DataArray(
        utc_datetimes,
        dims="time",
        name="time",
        # attrs={
        #     "standard_name": "time",
        #     "long_name": "Time",
        #     "axis": "T",
        #     "calendar": "gregorian",
        #     "timezone": "UTC",
        # },
    )

    # Drop old time variables and assign new coordinate
    ds = ds.drop_vars("Times")
    if "time" in ds:
        ds = ds.drop_vars("time")
    ds = ds.assign_coords(time=new_time)

    # Validate time is monotonic and has no duplicates
    time_series = pd.Series(ds.time.values)
    if not time_series.is_monotonic_increasing:
        print("Warning: Time values are not monotonically increasing. Sorting...")
        ds = ds.sortby("time")

    if time_series.duplicated().any():
        print("Warning: Duplicate timestamps found. Keeping first occurrence...")
        _, unique_indices = np.unique(ds.time.values, return_index=True)
        ds = ds.isel(time=unique_indices)

    return ds


def validate_and_standardize_lat_lon(ds):
    """
    Validates and standardizes latitude and longitude coordinates according to CF conventions.

    Parameters:
    ds (xarray.Dataset): Input dataset

    Returns:
    xarray.Dataset: Dataset with standardized lat/lon coordinates
    """
    # Handle latitude
    lat_names = ["latc", "lat", "latitude"]
    lon_names = ["lonc", "lon", "longitude"]

    # Find existing lat/lon coordinates
    existing_lat = next((name for name in lat_names if name in ds), None)
    existing_lon = next((name for name in lon_names if name in ds), None)

    if existing_lat:
        lat_data = ds[existing_lat]
        # Add CF standard attributes
        lat_attrs = {
            "standard_name": "latitude",
            "long_name": "Latitude",
            "units": "degrees_north",
            "axis": "Y",
            "valid_min": -90.0,
            "valid_max": 90.0,
        }

        # Create new latitude coordinate
        ds = ds.drop_vars(existing_lat)
        ds = ds.assign_coords(
            latitude=xr.DataArray(lat_data.values, dims=lat_data.dims, attrs=lat_attrs)
        )

        # Validate latitude values
        if (ds.latitude < -90).any() or (ds.latitude > 90).any():
            raise ValueError("Latitude values must be between -90 and 90 degrees")

    if existing_lon:
        lon_data = ds[existing_lon]
        # Add CF standard attributes
        lon_attrs = {
            "standard_name": "longitude",
            "long_name": "Longitude",
            "units": "degrees_east",
            "axis": "X",
            "valid_min": -180.0,
            "valid_max": 180.0,
        }

        # Create new longitude coordinate
        ds = ds.drop_vars(existing_lon)
        ds = ds.assign_coords(
            longitude=xr.DataArray(lon_data.values, dims=lon_data.dims, attrs=lon_attrs)
        )

        # Validate longitude values and wrap to [-180, 180]
        ds["longitude"] = xr.where(ds.longitude > 180, ds.longitude - 360, ds.longitude)

    return ds


def validate_and_calculate_mean_velocity(ds):
    """
    Validates velocity components and calculates mean velocity across depth layers.

    Parameters:
    ds (xarray.Dataset): Input dataset

    Returns:
    xarray.Dataset: Dataset with mean velocity calculations
    """
    required_vars = ["u", "v"]
    for var in required_vars:
        if var not in ds:
            raise ValueError(f"Missing required velocity component: {var}")

    # Calculate mean velocities across depth
    ds["u_avg"] = ds.u.mean(dim="siglay")
    ds["u_avg"].attrs.update(
        {
            "standard_name": "eastward_sea_water_velocity",
            "long_name": "Average Eastward Water Velocity",
            "units": "m s-1",
        }
    )

    ds["v_avg"] = ds.v.mean(dim="siglay")
    ds["v_avg"].attrs.update(
        {
            "standard_name": "northward_sea_water_velocity",
            "long_name": "Average Northward Water Velocity",
            "units": "m s-1",
        }
    )

    # Calculate vertical shear (veer)
    if "siglay" in ds.dims:
        u_diff = ds.u.diff("siglay")
        v_diff = ds.v.diff("siglay")
        depth_diff = ds.siglay.diff("siglay")

        ds["veer"] = np.sqrt(u_diff**2 + v_diff**2) / np.abs(depth_diff)
        ds["veer"].attrs.update(
            {
                "standard_name": "vertical_velocity_shear",
                "long_name": "Vertical Velocity Shear",
                "units": "s-1",
            }
        )

    # Calculate flow direction
    ds["flow_direction"] = np.rad2deg(np.arctan2(ds.v_avg, ds.u_avg))
    ds["flow_direction"].attrs.update(
        {
            "standard_name": "flow_direction",
            "long_name": "Flow Direction",
            "units": "degree",
            "valid_min": -180.0,
            "valid_max": 180.0,
        }
    )

    return ds


def validate_and_calculate_current_speed(ds):
    """
    Validates and calculates current speed from velocity components.

    Parameters:
    ds (xarray.Dataset): Input dataset

    Returns:
    xarray.Dataset: Dataset with current speed calculations
    """
    required_vars = ["u_avg", "v_avg"]
    for var in required_vars:
        if var not in ds:
            raise ValueError(f"Missing required mean velocity component: {var}")

    # Calculate current speed
    ds["current_speed"] = np.sqrt(ds["u_avg"] ** 2 + ds["v_avg"] ** 2)
    ds["current_speed"].attrs.update(
        {
            "standard_name": "sea_water_speed",
            "long_name": "Current Speed",
            "units": "m s-1",
            "type": "calculated",
        }
    )

    return ds


def validate_and_calculate_tidal_power_density(ds):
    """
    Validates and calculates tidal power density from current speed.

    Parameters:
    ds (xarray.Dataset): Input dataset

    Returns:
    xarray.Dataset: Dataset with tidal power density calculations
    """
    if "current_speed" not in ds:
        raise ValueError("Missing required variable: current_speed")

    # Calculate tidal power density (P = 0.5 * ρ * v³)
    seawater_density = 1025.0  # kg/m³
    ds["tidal_power_density"] = 0.5 * seawater_density * ds["current_speed"] ** 3
    ds["tidal_power_density"].attrs.update(
        {
            "standard_name": "tidal_power_density",
            "long_name": "Tidal Power Density",
            "units": "W m-2",
            "type": "calculated",
            "density_used": seawater_density,
        }
    )

    return ds


# This script builds an NetCDF4 file with the following 4 datasets
# - time_index (num_timesteps=8760, 1) - holds string timestamps at hour intervals (365*24 per year)
# - coordinates (num_points, 2) - lat column, lon column, converted from UTM when necessary
# - current_speed (num_timesteps, num_points) - sqrt(u^2 + v^2)
#       - every row is the total solution at that timestamp
#       - every column is a single points solution for the entire year timeseries
#       - Unit is m/s
# - tidal_power_density (num_timesteps, num_points) - 0.5*rho*current_speed^3 =  0.5*rho*sqrt(u^2 + v^2)^3
#   - Unit is W/m^2

# base_dir = "/projects/hindcastra/Tidal/Western_Passage_corrected"

# path_to_files = []
# path_to_files.extend(sorted(glob.glob(f"{base_dir}/01_Jan_Mar/*.nc")))
# path_to_files.extend(sorted(glob.glob(f"{base_dir}/02_Apr_Jun/*.nc")))
# path_to_files.extend(sorted(glob.glob(f"{base_dir}/03_Jul_Sep/*.nc")))
# path_to_files.extend(sorted(glob.glob(f"{base_dir}/04_Oct_Dec/*.nc")))

# # .to_numpy() extracts  np datetime64[ns] values from datetime conversion,
# # necessary for comparing and filtering by these dates
# start_date = pd.to_datetime("2017-01-01", utc=True).to_numpy()
# end_date = pd.to_datetime("2018-12-13 23:59:59", utc=True).to_numpy()

# UTM_ZONE = 19
# UTM_NORTHERN = True

# first_ds = xr.open_dataset(path_to_files[0], decode_times=False)
# LAT = first_ds["latc"]
# LON = first_ds["lonc"]

# if np.allclose(LAT, 0.0):
#     xc = first_ds["xc"]
#     yc = first_ds["yc"]

#     LAT, LON = utm.to_latlon(xc, yc, UTM_ZONE, northern=UTM_NORTHERN)


# # Get the current date and time
# current_datetime = datetime.now()

# # Format it as a string (e.g., "2024-08-14_15-30-00")
# formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")


# nc_filename = f"/scratch/asimms/Western_Passage_corrected.half_hourly.current_speed-tidal_power_density_fixed_ts_{formatted_datetime}.nc"

# output_ds = None


def calculate_tidal_qoi(spec):
    output_filepath = spec["output_filepath"]
    start_date = spec["filter_start_date"]
    end_date = spec["filter_end_date"]
    calculated_lat = None
    calculated_lon = None
    if "lat" in spec:
        calculated_lat = spec["lat"]
    if "lon" in spec:
        calculated_lon = spec["lon"]

    output_ds = None

    for this_path in spec["input_files"]:
        print(f"Working {this_path}...")
        ds = xr.open_dataset(this_path, decode_times=False)
        # ds = standardize_time(ds)
        # ds = standardize_lat_lon(ds)
        # ds = calculate_mean_velocity(ds)
        # ds = calculate_current_speed(ds)
        # ds = calculate_tidal_power_density(ds)

        # Use `Times` in place of `time`
        times_string = [ts.decode("utf-8") for ts in ds["Times"].values]
        # Extract datetime64[ns] values from datetime conversion
        utc_datetimes = pd.to_datetime(times_string, utc=True).values
        print(f"\tFilter start_date: {start_date}")
        print(f"\tFilter end_date: {end_date}")
        print(f"\tThis Start time: {utc_datetimes[0]}")
        print(f"\tThis End time: {utc_datetimes[-1]}")
        print(f"\tThis Number of timestamps: {len(utc_datetimes)}")
        timestamp_diff = pd.Series(utc_datetimes).diff()
        mean_timestamp_diff = timestamp_diff.mean()
        max_timestamp_diff = timestamp_diff.max()
        min_timestamp_diff = timestamp_diff.min()
        print(
            f"\tTimestamps are monotonic increasing: {pd.Series(utc_datetimes).is_monotonic_increasing}"
        )
        print(f"\tmean timestamp diff: {mean_timestamp_diff}")
        print(f"\tmax timestamp diff: {max_timestamp_diff}")
        print(f"\tmin timestamp diff: {min_timestamp_diff}")

        new_time = xr.DataArray(
            utc_datetimes,
            dims="time",
            name="time",
            attrs={
                "long_name": "Time",
                "standard_name": "time",
                "timezone": "UTC",
                # ValueError: failed to prevent overwriting existing key units in attrs on variable 'time'. This is probably an encoding field used by xarray to describe how a variable is serialized. To proceed, remove this key from the variable's attributes manually.
                # "units": "seconds since 1970-01-01 00:00:00",
            },
        )
        ds = ds.drop_vars("time")
        ds = ds.assign_coords(time=new_time)

        if calculated_lat is not None and calculated_lon is not None:
            ds["latc"] = calculated_lat
            ds["lonc"] = calculated_lon

        print("\tComputing u mean...")
        ds["u_avg"] = ds.u.mean(dim="siglay")
        ds["u_avg"].attrs["long_name"] = "Average Eastward Water Velocity"
        ds["u_avg"].attrs["units"] = "meters s-1"

        print("\tComputing v mean...")
        ds["v_avg"] = ds.v.mean(dim="siglay")
        ds["v_avg"].attrs["long_name"] = "Average Northward Water Velocity"
        ds["v_avg"].attrs["units"] = "meters s-1"

        print("\tComputing current speed...")
        ds["current_speed"] = np.sqrt(ds["u_avg"] ** 2 + ds["v_avg"] ** 2)
        ds["current_speed"].attrs = {
            "long_name": "Current Speed",
            "standard_name": "current_speed",
            "units": "meters s-1",
            "type": "calculated",
        }

        print("\tComputing tidal_density...")
        ds["tidal_power_density"] = 0.5 * 1025.0 * ds["current_speed"] ** 3
        ds["tidal_power_density"].attrs = {
            "long_name": "Tidal Power Density",
            "standard_name": "tidal_power_density",
            "units": "W/m^2",
            "type": "calculated",
        }

        subset_columns = ["current_speed", "tidal_power_density"]
        subset_vars = {var: ds[var] for var in subset_columns if var in ds}
        subset_ds = xr.Dataset(subset_vars)

        for var in subset_columns:
            if var in ds:
                subset_ds[var].attrs.update(ds[var].attrs)

        subset_ds.attrs = ds.attrs
        time_values = subset_ds["time"].values

        # Check time is valid and makes sense
        # Step 1: Check if 'time' is in order
        is_in_order = (
            pd.Series(time_values) == pd.Series(time_values).sort_values()
        ).all()

        # Step 2: Check for duplicates
        has_duplicates = pd.Series(time_values).duplicated().any()

        # Step 3: If if we have duplicates, fix them
        if has_duplicates:
            print("Time Duplicates found, keeping the first occurrence.")
            print("Duplicates:", pd.Series(time_values).duplicated())
            # Remove duplicates, keeping the first occurrence
            _, unique_indices = np.unique(time_values, return_index=True)
            subset_ds = ds.isel(time=unique_indices)

        # Step 4: If time is not in order, fix it
        if not is_in_order:
            print("Timestamps were out of order. Dataset sorting...")
            subset_ds = subset_ds.sortby("time")

        if has_duplicates is False and is_in_order is True:
            print("Subset ds timestamps have no duplicates and are ordered correctly")

        print(subset_ds.time.values)
        print(subset_ds.time.values[0])
        print(start_date)
        print(end_date)
        print(type(subset_ds.time.values))
        print(type(subset_ds.time.values[0]))
        print(type(start_date))
        print(type(end_date))
        # Filter the subset_ds to only include data from 2017
        time_filter = (subset_ds.time >= start_date) & (subset_ds.time < end_date)
        subset_ds_filtered = subset_ds.sel(time=time_filter)

        if output_ds is None:
            output_ds = subset_ds
        else:
            # Before concatenating, remove duplicates from combined_ds based on 'time'
            existing_times = output_ds.time.values
            new_times = subset_ds.time.values

            # Identify duplicates
            existing_times_df = pd.DataFrame(existing_times, columns=["time"])
            new_times_df = pd.DataFrame(new_times, columns=["time"])
            duplicates = new_times_df[
                new_times_df["time"].isin(existing_times_df["time"])
            ]

            if not duplicates.empty:
                print(
                    f"Skipping the following duplicate timestamps found in {this_path}:"
                )
                print(duplicates)

            # Remove duplicates from subset_ds
            # Create a boolean mask for non-duplicate times
            mask = ~subset_ds.time.isin(duplicates["time"].values)

            # Select the data where the mask is True
            subset_ds_cleaned = subset_ds.sel(time=mask)

            # Concatenate cleaned subset_ds to the combined_ds
            output_ds = xr.concat([output_ds, subset_ds_cleaned], dim="time")
            timestamp_series = pd.Series(output_ds["time"])
            timestamp_diff = timestamp_series.diff()
            timestamp_diff = timestamp_diff
            mean_timestamp_diff = timestamp_diff.mean()
            max_timestamp_diff = timestamp_diff.max()
            min_timestamp_diff = timestamp_diff.min()
            print(
                f"\tOutput ds Timestamps are monotonic increasing: {timestamp_series.is_monotonic_increasing}"
            )
            print(f"\tOutput ds mean timestamp diff: {mean_timestamp_diff}")
            print(f"\tOutput ds max timestamp diff: {max_timestamp_diff}")
            print(f"\tOutput ds min timestamp diff: {min_timestamp_diff}")
            print(output_ds.info())
            print(output_ds.time)
            time_filter = (output_ds.time >= start_date) & (output_ds.time < end_date)
            output_ds = output_ds.sel(time=time_filter)
            print(output_ds.info())
            print(f"Saving to {output_filepath}")
            output_ds.to_netcdf(output_filepath)

    if output_ds is not None:
        time_filter = (output_ds.time >= start_date) & (output_ds.time < end_date)
        output_ds = output_ds.sel(time=time_filter)
        print(output_ds.info())
        print(f"Saving to {output_filepath}")
        output_ds.to_netcdf(output_filepath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "location", type=str, help="Folder name where Tidal nc files are located"
    )
    args = parser.parse_args()

    this_location = args.location

    dispatch = {
        "Western_Passage_corrected": generate_western_passage_corrected_spec,
        "Puget_Sound_corrected": generate_puget_sound_corrected_spec,
        "Cook_Inlet_PNNL": generate_cook_inlet_spec,
        "PIR_full_year": generate_pir_spec,
        "Aleutian_Islands_year": generate_aleutian_spec,
    }

    if this_location not in dispatch.keys():
        print("Please input a valid location. Valid options are:")
        [print(key) for key in dispatch.keys()]
        exit(1)

    calculate_tidal_qoi(dispatch[this_location]())
