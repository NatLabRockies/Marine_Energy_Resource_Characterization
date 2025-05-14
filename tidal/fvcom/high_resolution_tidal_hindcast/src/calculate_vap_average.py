from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from . import attrs_manager, file_manager, file_name_convention_manager, nc_manager


def verify_timestamps(nc_files, expected_timestamps, expected_delta_t_seconds, config):
    """Verify timestamp integrity across all files before processing."""

    total_timestamps = 0
    last_timestamp = None
    expected_diff = pd.Timedelta(seconds=expected_delta_t_seconds)

    for nc_file in nc_files:
        # Open dataset with minimal loading
        ds = nc_manager.nc_open(nc_file, config, decode_times=True)
        times = ds.time.values

        # Check count
        total_timestamps += len(times)

        # Check spacing within file
        time_diffs = pd.Series(times).diff()
        if not all(time_diffs[1:] == expected_diff):
            raise ValueError(f"Irregular timestamp spacing in {nc_file}")

        # Check continuity between files
        if last_timestamp is not None:
            gap = pd.Timestamp(times[0]) - last_timestamp
            if gap != expected_diff:
                raise ValueError(f"Gap between files: {nc_file}")

        last_timestamp = pd.Timestamp(times[-1])
        ds.close()

    if total_timestamps != expected_timestamps:
        raise ValueError(
            f"Expected {expected_timestamps} timestamps, found {total_timestamps}"
        )

    return True


def verify_constant_variables(ds1, ds2, constant_vars):
    """
    Verify that specified variables have the same values across datasets.

    Args:
        ds1: First dataset containing the variables
        ds2: Second dataset containing the variables
        constant_vars: List of variable names to verify

    Returns:
        dict: Dictionary of variables with comparison results
    """
    results = {}

    for var_name in constant_vars:
        if var_name in ds1 and var_name in ds2:
            # Check if the values are identical
            if np.array_equal(ds1[var_name].values, ds2[var_name].values):
                results[var_name] = True
            else:
                results[var_name] = False
        else:
            # Variable missing in one of the datasets
            results[var_name] = False

    return results


class VAPSummaryCalculator:
    """
    Class for calculating averages, max values, and 95th percentiles across VAP NetCDF files.
    Handles both yearly and monthly processing with a unified approach.
    """

    def __init__(self, config, location_name):
        self.config = config
        self.location_name = location_name
        self.location = config["location_specification"][location_name]
        self.constant_variables = ["nv"]  # Variables that should remain constant

        # Common paths
        self.vap_path = file_manager.get_vap_output_dir(config, self.location)
        self.vap_nc_files = sorted(list(self.vap_path.rglob("*.nc")))

        # Verify minimum files
        if len(self.vap_nc_files) < 12:
            raise ValueError(
                f"Expecting at least 12 files in {self.vap_path}, found {len(self.vap_nc_files)}"
            )

        # Calculate expected timestamps
        self.days_per_year = 365
        if self.location["output_name"] == "WA_puget_sound":
            # Puget sound is missing one day
            self.days_per_year = 364

        self.seconds_per_year = self.days_per_year * 24 * 60 * 60
        self.expected_timestamps = int(
            self.seconds_per_year / self.location["expected_delta_t_seconds"]
        )

        # Track max values across all files for each variable
        self.max_values = {}

        # Identify variable types
        self.avg_vars = []  # Regular variables for averaging
        self.max_vars = []  # Max variables
        self.p95_vars = []  # 95th percentile variables

    def _verify_timestamps(self):
        print("Verifying timestamps across all files...")
        verify_timestamps(
            self.vap_nc_files,
            self.expected_timestamps,
            self.location["expected_delta_t_seconds"],
            self.config,
        )

    def _verify_constant_vars(self, ds, first_ds):
        """Verify constant variables haven't changed."""
        for var in self.constant_variables:
            if var in list(ds.keys()) and var in list(first_ds.keys()):
                if not np.array_equal(ds[var].values, first_ds[var].values):
                    print(
                        f"WARNING: Variable '{var}' differs between files. Using value from first file."
                    )

    def _initialize_dataset(self, dataset, constant_variables):
        """
        Initialize a template dataset and identify variable types.
        """
        # Initialize with the first timestep
        template = dataset.isel(time=0).copy(deep=True)
        first_timestamp = dataset.time.isel(time=0).values
        time_attrs = dataset.time.attrs.copy()

        # Store all variable attributes
        var_attrs = {}
        for var in dataset.data_vars:
            var_attrs[var] = dataset[var].attrs.copy()

            # Identify variables by type
            if "time" in dataset[var].dims and var not in constant_variables:
                if "water_column_max" in var:
                    self.max_vars.append(var)
                elif "water_column_95th_percentile" in var:
                    self.p95_vars.append(var)
                else:
                    self.avg_vars.append(var)

        # Initialize variables with appropriate data types
        for var in template.data_vars:
            if "time" in dataset[var].dims and var not in constant_variables:
                # Use float64 for better numerical stability
                if np.issubdtype(template[var].dtype, np.integer):
                    template[var] = template[var].astype(np.float64)
                elif np.issubdtype(template[var].dtype, np.floating):
                    template[var] = template[var].astype(np.float64)

                # Initialize to zero (will be properly set during processing)
                template[var].values.fill(0)

                # Initialize max_values tracking for max variables
                if var in self.max_vars:
                    self.max_values[var] = None

        return template, first_timestamp, time_attrs, var_attrs

    def _update_averages(self, result_ds, dataset, count):
        """
        Update running averages for regular variables only.
        """
        current_times = len(dataset.time)

        # Update running average for regular variables only
        for var in self.avg_vars:
            if var in dataset.data_vars:
                # Calculate mean for this dataset
                if np.issubdtype(dataset[var].dtype, np.integer):
                    # Safely convert integers to float64 first
                    file_avg = dataset[var].astype(np.float64).mean(dim="time")
                else:
                    file_avg = dataset[var].mean(dim="time")

                # Update the rolling average using the formula:
                # new_avg = old_avg + (new_value - old_avg) / new_count
                if count == 0:
                    # First iteration - simply set to the file average
                    result_ds[var] = file_avg
                else:
                    weight = current_times / (count + current_times)
                    result_ds[var] = (
                        result_ds[var] + (file_avg - result_ds[var]) * weight
                    )

        return count + current_times

    def _update_max_values(self, result_ds, dataset, count):
        """
        Update max values and track them for 95th percentile calculation.
        """
        # Process each max variable
        for var in self.max_vars:
            if var in dataset.data_vars:
                # Calculate the max for this dataset
                file_max = dataset[var].max(dim="time")

                # Track all max values for later 95th percentile calculation
                if self.max_values[var] is None:
                    self.max_values[var] = file_max.expand_dims(dim={"file": [count]})
                else:
                    # Add to our collection of max values
                    new_max = file_max.expand_dims(dim={"file": [count]})
                    self.max_values[var] = xr.concat(
                        [self.max_values[var], new_max], dim="file"
                    )

                # For the running result, take the maximum of the current max and the previous max
                if count == 0:
                    # First iteration - simply set to the file max
                    result_ds[var] = file_max
                else:
                    # Update the max using element-wise maximum
                    result_ds[var] = xr.where(
                        file_max > result_ds[var], file_max, result_ds[var]
                    )

        return result_ds

    def _calculate_percentiles(self, result_ds):
        """
        Calculate 95th percentiles of the max values after all files have been processed.
        """
        print("Calculating 95th percentiles of max values...")

        # Map each p95 variable to its corresponding max variable
        max_to_p95_map = {}
        for p95_var in self.p95_vars:
            # Find the corresponding max variable by replacing '95th_percentile' with 'max'
            base_name = p95_var.replace("_95th_percentile_", "_max_")
            if base_name in self.max_vars:
                max_to_p95_map[base_name] = p95_var

        # Calculate 95th percentile for each p95 variable
        for max_var, p95_var in max_to_p95_map.items():
            if max_var in self.max_values and self.max_values[max_var] is not None:
                # Calculate the 95th percentile along the file dimension
                p95_value = self.max_values[max_var].quantile(0.95, dim="file")
                result_ds[p95_var] = p95_value
                print(f"Calculated 95th percentile for {p95_var} from {max_var}")

        return result_ds

    def _process_files(self, ds_template, file_list, time_filter=None):
        """
        Process a list of files, updating averages, max values, and 95th percentiles.

        Args:
            ds_template: Template dataset initialized with proper dimensions
            file_list: List of files to process
            time_filter: Optional function to filter timestamps in each dataset

        Returns:
            tuple: (updated_ds, first_timestamp, time_attrs, var_attrs, source_files)
        """
        result_ds = ds_template
        count = 0
        source_files = []
        first_timestamp = None
        time_attrs = None
        var_attrs = {}
        first_ds = None

        # Process each file
        for i, nc_file in enumerate(file_list):
            print(f"Processing File {i}: {nc_file}")
            ds = nc_manager.nc_open(nc_file, self.config)

            # Apply time filter if provided
            if time_filter is not None:
                ds_filtered = time_filter(ds)
                if ds_filtered is None or len(ds_filtered.time) == 0:
                    ds.close()
                    continue  # Skip if no data after filtering
                ds = ds_filtered

            # Initialize if this is the first valid file
            if first_ds is None:
                result_ds, first_timestamp, time_attrs, var_attrs = (
                    self._initialize_dataset(ds, self.constant_variables)
                )
                first_ds = ds.copy()
            else:
                self._verify_constant_vars(ds, first_ds)

            # Update averages and max values separately
            count = self._update_averages(result_ds, ds, count)
            result_ds = self._update_max_values(result_ds, ds, count)

            # Track source files
            if str(nc_file) not in source_files:
                source_files.append(str(nc_file))

            ds.close()

        # Calculate 95th percentiles after processing all files
        if first_ds is not None:  # Only if we processed at least one file
            result_ds = self._calculate_percentiles(result_ds)

        return result_ds, first_timestamp, time_attrs, var_attrs, source_files

    def _finalize_dataset(self, running_avg, first_timestamp, time_attrs, var_attrs):
        """
        Finalize the averaged dataset by restoring attributes and metadata.

        Args:
            running_avg: Running average dataset
            first_timestamp: First timestamp for time coordinate
            time_attrs: Time dimension attributes
            var_attrs: Variable attributes dictionary

        Returns:
            xarray.Dataset: Finalized dataset with restored attributes
        """
        # Restore variable attributes and potentially convert back to original types
        for var in running_avg.data_vars:
            if (
                var not in running_avg.dims
                and var not in running_avg.coords
                and var not in self.constant_variables
            ):
                # Restore variable attributes
                if var in var_attrs:
                    running_avg[var].attrs = var_attrs[var]

                    # Optionally convert back to original type if safe and configured
                    original_dtype = var_attrs[var].get("_original_dtype")
                    if original_dtype and self.config.get("preserve_dtypes", False):
                        try:
                            # Test conversion for safety
                            test_conversion = running_avg[var].values.astype(
                                original_dtype
                            )
                            running_avg[var] = running_avg[var].astype(original_dtype)
                        except (OverflowError, ValueError):
                            print(
                                f"WARNING: Cannot safely convert {var} back to {original_dtype}. Keeping higher precision."
                            )

        # Restore the time coordinate with its attributes
        running_avg = running_avg.assign_coords(time=("time", [first_timestamp]))
        running_avg["time"].attrs = time_attrs  # Restore time attributes

        return running_avg

    def _save_dataset(
        self,
        dataset,
        output_path,
        filename,
        source_files,
        data_level,
        temporal="average",
    ):
        """
        Save the dataset to a NetCDF file with proper attributes and encoding.

        Args:
            dataset: xarray Dataset to save
            output_path: Directory path to save to
            filename: Base filename
            source_files: List of source files used to create this dataset
            data_level: Data level identifier
            temporal: Temporal specification for filename

        Returns:
            Path: Path to the saved file
        """
        # Generate output filename
        data_level_file_name = (
            file_name_convention_manager.generate_filename_for_data_level(
                dataset,
                self.location["output_name"],
                self.config["dataset"]["name"],
                data_level,
                temporal=temporal,
            )
        )

        # Add standard attributes
        dataset = attrs_manager.standardize_dataset_global_attrs(
            dataset,
            self.config,
            self.location,
            data_level,
            source_files,
        )

        file_path = Path(output_path, filename.format(data_level_file_name))
        print(f"Saving to {file_path}...")

        nc_manager.nc_write(dataset, file_path, self.config)

        return file_path

    def calculate_yearly_average(self):
        """
        Calculate yearly average values, max values, and 95th percentiles across VAP NC files.
        """
        output_path = file_manager.get_yearly_summary_vap_output_dir(
            self.config, self.location
        )
        output_nc_files = list(output_path.rglob("*.nc"))

        if len(output_nc_files) > 0:
            print(
                f"{len(output_nc_files)} summary files already exist. Skipping yearly averaging."
            )
            return None

        # Verify timestamps
        self._verify_timestamps()

        # Reset variable tracking for this calculation
        self.avg_vars = []
        self.max_vars = []
        self.p95_vars = []
        self.max_values = {}

        # Process all files
        print(f"Starting yearly processing of {len(self.vap_nc_files)} vap files...")
        result_ds, first_timestamp, time_attrs, var_attrs, source_files = (
            self._process_files(None, self.vap_nc_files)
        )

        if result_ds is None:
            print("No data found for yearly processing")
            return None

        print("Finalizing yearly dataset...")
        # Finalize the dataset
        finalized_ds = self._finalize_dataset(
            result_ds, first_timestamp, time_attrs, var_attrs
        )

        print(finalized_ds.info())
        print(finalized_ds.time)

        # Save the yearly average
        return self._save_dataset(
            finalized_ds,
            output_path,
            "001.{}",
            source_files,
            "b3",
            temporal="1_year_average",
        )

    def calculate_monthly_averages(self):
        """
        Calculate monthly average values, max values, and 95th percentiles across VAP NC files.
        """
        output_path = file_manager.get_monthly_summary_vap_output_dir(
            self.config, self.location
        )
        # Create the output directory if it doesn't exist
        output_path.mkdir(parents=True, exist_ok=True)

        # Check if monthly files already exist
        output_nc_files = list(output_path.rglob("*.nc"))
        if len(output_nc_files) >= 12:
            print(
                f"{len(output_nc_files)} monthly summary files already exist. Skipping monthly averaging."
            )
            return output_path

        # Verify timestamps
        self._verify_timestamps()

        # Load all datasets to get the full time dimension
        print("Loading all datasets to extract timestamps...")
        all_times = []
        for nc_file in self.vap_nc_files:
            ds = nc_manager.nc_open(nc_file, self.config)
            all_times.append(ds.time.values)
            ds.close()

        # Flatten the list of arrays into a single array and convert to pandas for easier handling
        all_timestamps = pd.to_datetime(np.concatenate(all_times))

        # Group timestamps by month
        months = {}
        for timestamp in all_timestamps:
            month_key = (timestamp.year, timestamp.month)
            if month_key not in months:
                months[month_key] = []
            months[month_key].append(timestamp)

        print(f"Found data for {len(months)} distinct months")

        # Process each month separately
        for month_key, month_timestamps in months.items():
            year, month = month_key
            month_name = pd.Timestamp(year=year, month=month, day=1).strftime("%B")
            print(
                f"Processing {month_name} {year} with {len(month_timestamps)} timestamps"
            )

            # Convert back to numpy datetime64 for comparison with xarray
            month_timestamps_np = np.array(
                [np.datetime64(ts) for ts in month_timestamps]
            )

            # Define a time filter function for this month
            def month_filter(ds):
                month_mask = np.isin(ds.time.values, month_timestamps_np)
                if not any(month_mask):
                    return None
                return ds.isel(time=month_mask)

            # Reset variable tracking for this month
            self.avg_vars = []
            self.max_vars = []
            self.p95_vars = []
            self.max_values = {}

            # Process files for this month
            result_ds, first_timestamp, time_attrs, var_attrs, source_files = (
                self._process_files(None, self.vap_nc_files, time_filter=month_filter)
            )

            # Skip if no data was found for this month
            if result_ds is None:
                print(f"No data found for {month_name} {year}, skipping")
                continue

            # Finalize the dataset
            monthly_ds = self._finalize_dataset(
                result_ds, first_timestamp, time_attrs, var_attrs
            )

            # Additional monthly metadata
            monthly_ds.attrs["month"] = month
            monthly_ds.attrs["year"] = year
            monthly_ds.attrs["month_name"] = month_name

            # Save the monthly average
            month_str = f"{year}_{month:02d}"
            self._save_dataset(
                monthly_ds,
                output_path,
                f"{month:02d}.{{}}",
                source_files,
                "b2",
                temporal=f"{month_str}_average",
            )

        return output_path


def calculate_vap_yearly_average(config, location):
    """Calculate yearly averages, max values, and 95th percentiles for VAP variables."""
    averager = VAPSummaryCalculator(config, location)
    return averager.calculate_yearly_average()


def calculate_vap_monthly_average(config, location):
    """Calculate monthly averages, max values, and 95th percentiles for VAP variables."""
    averager = VAPSummaryCalculator(config, location)
    return averager.calculate_monthly_averages()
