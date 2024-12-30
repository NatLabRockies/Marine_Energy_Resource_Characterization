from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from . import coord_manager, file_manager, time_manager


class TimeVerifier:
    def __init__(self):
        self.original_time = []
        self.pandas_time = []
        self.unix_ns_time = []
        self.source_file = []

    def verify_individual_dataset(self, ds, expected_delta_t_seconds, filepath):
        this_times = time_manager.standardize_fvcom_time(ds)

        for key in this_times.keys():
            time_array = this_times[key]
            if self.does_time_always_increase(time_array) is False:
                raise ValueError(
                    f"Time verification failure in {filepath}. Time is not always increasing. Check timestamps below\n{str(time_array)}"
                )

        if (
            self.verify_frequency(this_times["Timestamp"], expected_delta_t_seconds)
            is False
        ):
            print(this_times["Timestamp"])
            print(pd.Series(this_times["Timestamp"]).diff())
            raise ValueError(
                f"Time verification failure in {filepath}. Delta t is different than {expected_delta_t_seconds} seconds. Check timestamps below\n{str(this_times['Timestamp'])}"
            )

        num_timestamps = len(this_times["Timestamp"])
        self.original_time.extend(this_times["original"])
        self.pandas_time.extend(this_times["Timestamp"])
        self.unix_ns_time.extend(this_times["datetime64[ns]"])
        self.source_file.extend([filepath] * num_timestamps)

    def does_time_always_increase(self, time_array):
        return time_manager.does_time_always_increase(time_array)

    def verify_frequency(self, pandas_timestamp_series, expected_delta_t_seconds):
        time_delta_stats = time_manager.calculate_time_delta_seconds(
            pandas_timestamp_series
        )
        return time_delta_stats["mean_dt"] == expected_delta_t_seconds

    def create_valid_timestamps_df(self):
        return pd.DataFrame(
            {
                "original": self.original_time,
                "timestamp": self.pandas_time,
                "time_ns": self.pandas_time,
                "source_file": [str(f) for f in self.source_file],
            }
        )

    def verify_complete_dataset(self, location):
        """Verifies time series integrity for entire dataset"""
        # Convert to pandas timestamps and remove duplicates while preserving order
        timestamps = pd.to_datetime(pd.Series(self.pandas_time)).drop_duplicates(
            keep="first"
        )
        timestamps = timestamps.sort_values()

        if not all(
            pd.to_datetime(self.pandas_time) == pd.to_datetime(sorted(self.pandas_time))
        ):
            raise ValueError("Original timestamps were not in chronological order")

        start_date = pd.to_datetime(location["start_date"], utc=True)
        end_date = pd.to_datetime(location["end_date"], utc=True)

        if timestamps.iloc[0] > start_date:
            raise ValueError(
                f"Dataset starts at {timestamps.iloc[0]}, expected {start_date}"
            )
        if timestamps.iloc[-1] < end_date:
            raise ValueError(
                f"Dataset ends at {timestamps.iloc[-1]}, expected {end_date}"
            )

        time_deltas = timestamps.diff()[1:]  # Skip first NaN delta
        expected_delta = pd.Timedelta(seconds=location["expected_delta_t_seconds"])

        if not all(delta == expected_delta for delta in time_deltas):
            incorrect_deltas = [
                (t1, t2)
                for t1, t2, delta in zip(timestamps[:-1], timestamps[1:], time_deltas)
                if delta != expected_delta
            ]
            print(self.pandas_time)
            print(time_deltas)
            print(expected_delta)
            raise ValueError(f"Inconsistent time steps found: {incorrect_deltas}")

        return True


class CoordinateSystemVerifier:
    def __init__(self):
        self.lat_centers = []
        self.lon_centers = []
        self.lat_corners = []
        self.lon_corners = []
        self.reference_dataset = None

    def verify_individual_dataset(self, ds, location):
        utm_zone = None
        if location["coordinates"]["system"] == "utm":
            utm_zone = location["coordinates"]["zone"]

        coords = coord_manager.standardize_fvcom_coords(ds, utm_zone)

        if len(self.lat_centers) == 0:
            self.lat_centers = coords["lat_centers"]
            self.lon_centers = coords["lon_centers"]
            self.lat_corners = coords["lat_corners"]
            self.lon_corners = coords["lon_corners"]
            self.reference_dataset = location["output_name"]
            return True

        # Verify arrays match reference dataset
        matches = (
            np.array_equal(coords["lat_centers"], self.lat_centers)
            and np.array_equal(coords["lon_centers"], self.lon_centers)
            and np.array_equal(coords["lat_corners"], self.lat_corners)
            and np.array_equal(coords["lon_corners"], self.lon_corners)
        )

        if not matches:
            raise ValueError(
                f"Coordinates in dataset {location['output_name']} do not match reference dataset {self.reference_dataset}"
            )

        return matches


class GlobalAttributeEqualityVerifier:
    """Verifies dataset global attributes match the first dataset, with exclusions"""

    def __init__(self, exclude_keys=None):
        self.reference_attrs = None
        self.reference_dataset = None
        self.exclude_keys = set(exclude_keys or [])

    def get_filtered_attrs(self, ds):
        return {k: v for k, v in ds.attrs.items() if k not in self.exclude_keys}

    def verify_individual_dataset(self, ds, location):
        filtered_attrs = self.get_filtered_attrs(ds)

        if self.reference_attrs is None:
            self.reference_attrs = filtered_attrs
            self.reference_dataset = location["output_name"]
            return True

        if filtered_attrs != self.reference_attrs:
            mismatched = {
                k: (self.reference_attrs.get(k), filtered_attrs.get(k))
                for k in set(self.reference_attrs) | set(filtered_attrs)
                if k not in self.exclude_keys
                and self.reference_attrs.get(k) != filtered_attrs.get(k)
            }
            raise ValueError(
                f"Global attributes mismatch in {location['output_name']} vs {self.reference_dataset}:\n"
                f"Different values: {mismatched}"
            )
        return True


class DatasetStructureEqualityVerifier:
    """Verifies variable/dimension/coordinate names and their attributes match across datasets"""

    def __init__(self):
        self.reference_components = None  # List of (name, type, attrs) tuples
        self.reference_dataset = None

    def get_component_info(self, ds):
        components = []
        # Get variables, dimensions, coordinates info
        for name in sorted(list(ds.variables)):
            components.append((name, "variable", dict(ds[name].attrs)))
        for name in sorted(list(ds.dims)):
            components.append((name, "dimension", {}))  # Dimensions don't have attrs
        for name in sorted(list(ds.coords)):
            components.append((name, "coordinate", dict(ds[name].attrs)))
        return components

    def verify_individual_dataset(self, ds, location):
        current = self.get_component_info(ds)

        if self.reference_components is None:
            self.reference_components = current
            self.reference_dataset = location["output_name"]
            return True

        if current != self.reference_components:
            for ref, cur in zip(self.reference_components, current):
                if ref != cur:
                    raise ValueError(
                        f"Mismatch in {location['output_name']} for {ref[1]} {ref[0]}: "
                        f"Expected {ref}, got {cur}"
                    )
            # If lists are different lengths
            raise ValueError(
                f"Structure mismatch between {location['output_name']} and reference dataset {self.reference_dataset}"
            )
        return True


def model_specification_verifier(config, ds, filepath):
    # Dictionary of attributes to verify
    verifications = {
        "source": {
            "expected": config["model_specification"]["model_version"],
            "actual": ds.attrs["source"],
            "name": "Model",
        },
        "Conventions": {
            "expected": config["model_specification"]["conventions"],
            "actual": ds.attrs["Conventions"],
            "name": "Convention",
        },
    }

    # Check model and convention attributes
    for attr, check in verifications.items():
        if check["actual"] != check["expected"]:
            raise ValueError(
                f"{check['name']} {check['actual']} in {filepath} does not match "
                f"expected {check['name'].lower()} {check['expected']}"
            )

    # Verify required variables
    required_vars = config["model_specification"]["required_original_variables"]
    for var_name, expected_standard_name in required_vars.items():
        if var_name not in ds.variables:
            raise ValueError(f"Required variable {var_name} missing in {filepath}")

        var = ds[var_name]
        if "standard_name" not in var.attrs:
            raise ValueError(
                f"Variable {var_name} missing standard_name attribute in {filepath}"
            )

        if var.attrs["standard_name"] != expected_standard_name:
            raise ValueError(
                f"Variable {var_name} has standard_name {var.attrs['standard_name']}, "
                f"expected {expected_standard_name} in {filepath}"
            )


def verify_dataset(config, location, nc_files):
    tracking_folder = file_manager.get_tracking_output_dir(config)
    tracking_path = Path(
        tracking_folder, f"{location['output_name']}_verify_step_tracking.parquet"
    )

    # Check if tracking file exists
    if tracking_path.exists():
        print(f"\tDataset already verified: {location['output_name']}")
        return pd.read_parquet(tracking_path)

    time_verifier = TimeVerifier()
    coord_system_verifier = CoordinateSystemVerifier()
    global_attr_equal_verifier = GlobalAttributeEqualityVerifier(
        exclude_keys=["history"]
    )
    dataset_structure_equal_verifier = DatasetStructureEqualityVerifier()

    for nc_file in nc_files:
        print(f"\tVerifying {nc_file.name}...")
        ds = xr.open_dataset(nc_file, decode_times=False)

        model_specification_verifier(config, ds, nc_file)
        time_verifier.verify_individual_dataset(
            ds, location["expected_delta_t_seconds"], nc_file
        )
        coord_system_verifier.verify_individual_dataset(ds, location)
        global_attr_equal_verifier.verify_individual_dataset(ds, location)
        dataset_structure_equal_verifier.verify_individual_dataset(ds, location)

    time_verifier.verify_complete_dataset(location)

    timestamps_df = time_verifier.create_valid_timestamps_df()
    timestamps_df.to_parquet(tracking_path)

    return timestamps_df
