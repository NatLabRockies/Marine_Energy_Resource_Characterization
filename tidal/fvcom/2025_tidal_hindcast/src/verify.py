import numpy as np
import pandas as pd
import xarray as xr

from . import time_manager
from . import coord_manager


class TimeVerifier:
    def __init__(self):
        self.original_time = []
        self.pandas_time = []
        self.unix_ns_time = []

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
            raise ValueError(
                f"Time verification failure in {filepath}. Delta t is different than {expected_delta_t_seconds} seconds. Check timestamps below\n{str(this_times['Timestamp'])}"
            )

        self.original_time.extend(this_times["original"])
        self.pandas_time.extend(this_times["Timestamp"])
        self.unix_ns_time.extend(this_times["datetime64[ns]"])

    def does_time_always_increase(self, time_array):
        return time_manager.does_time_always_increase(time_array)

    def verify_frequency(self, pandas_timestamp_series, expected_delta_t_seconds):
        time_delta_stats = time_manager.calculate_time_delta_seconds(
            pandas_timestamp_series
        )
        return time_delta_stats["mean_dt"] == expected_delta_t_seconds

    def verify_complete_dataset(self, location):
        """Verifies time series integrity for entire dataset"""
        # Convert to pandas timestamps for easier manipulation
        timestamps = pd.to_datetime(set(self.pandas_time))  # Remove duplicates
        timestamps = sorted(timestamps)  # Sort chronologically

        # Check if original order matches sorted order
        if not all(
            pd.to_datetime(self.pandas_time) == pd.to_datetime(sorted(self.pandas_time))
        ):
            raise ValueError("Original timestamps were not in chronological order")

        # Verify time range
        start_date = pd.to_datetime(location["start_date"])
        end_date = pd.to_datetime(location["end_date"])

        if timestamps[0] > start_date:
            raise ValueError(
                f"Dataset starts at {timestamps[0]}, expected {start_date}"
            )
        if timestamps[-1] < end_date:
            raise ValueError(f"Dataset ends at {timestamps[-1]}, expected {end_date}")

        # Verify delta t
        time_deltas = np.diff(timestamps)
        expected_delta = pd.Timedelta(seconds=location["expected_delta_t_seconds"])

        if not all(delta == expected_delta for delta in time_deltas):
            incorrect_deltas = [
                (t1, t2)
                for t1, t2, delta in zip(timestamps[:-1], timestamps[1:], time_deltas)
                if delta != expected_delta
            ]
            raise ValueError(f"Inconsistent time steps found: {incorrect_deltas}")


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
    """Verifies all dataset global attributes match the first dataset"""

    def __init__(self):
        self.reference_attrs = None
        self.reference_dataset = None

    def verify_individual_dataset(self, ds, location):
        if self.reference_attrs is None:
            self.reference_attrs = dict(ds.attrs)
            self.reference_dataset = location["output_name"]
            return True

        if dict(ds.attrs) != self.reference_attrs:
            raise ValueError(
                f"Global attributes in {location['output_name']} do not match reference dataset {self.reference_dataset}"
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
    time_verifier = TimeVerifier()
    coord_system_verifier = CoordinateSystemVerifier()
    global_attr_equal_verifier = GlobalAttributeEqualityVerifier()
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
