import concurrent.futures
import gc
import os
import queue
import threading
import time

from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import xarray as xr

from . import attrs_manager, file_manager, file_name_convention_manager, time_manager


def partition_by_time(config, location_key, time_df, force_reprocess=False):
    """
    Partitions time-series data into separate files based on configured frequency.
    Skips processing if the expected number of output files already exists.
    """
    location = config["location_specification"][location_key]
    output_dir = file_manager.get_standardized_partition_output_dir(config, location)
    output_dir.mkdir(parents=True, exist_ok=True)
    partition_files = []

    # Get partition frequency from config and ensure timestamps are datetime objects
    freq = location["partition_frequency"]
    time_df["timestamp"] = pd.to_datetime(time_df["timestamp"])

    # Check how many existing files we have for this period number
    existing_files = list(output_dir.glob("*.nc"))

    files_to_be_generated_count = len(
        time_df.groupby(pd.Grouper(key="timestamp", freq=freq))
    )

    # Skip if we already have the expected number of files and aren't forcing reprocess
    if len(existing_files) == files_to_be_generated_count and not force_reprocess:
        print(
            f"{output_dir} already has {len(existing_files)} files, skipping time partitioning..."
        )
        return existing_files

    # Process data in time-based groups
    for count, (period_start, period_df) in enumerate(
        time_df.groupby(pd.Grouper(key="timestamp", freq=freq)), 1
    ):
        if period_df.empty:
            continue

        print(f"Processing period: {period_start} with frequency string {freq}")

        # Count how many unique source files we have for this period
        unique_std_files = period_df["std_files"].unique()

        datasets = []
        source_filenames = set()

        for std_file in unique_std_files:
            print(f"Adding {std_file} to {period_start} output dataset")

            print("Opening dataset...")
            ds = xr.open_dataset(std_file)

            # Track source filenames
            if "source_files" in ds.attrs:
                filenames = [Path(f).name for f in ds.attrs["source_files"]]
                source_filenames.update(filenames)

            # Get timestamps for this file and period
            file_timestamps = period_df[period_df["std_files"] == std_file][
                "timestamp"
            ].values
            print("file_timestamps:", file_timestamps)

            # Subset the dataset
            time_indices = np.isin(ds.time.values, file_timestamps)
            print("Subsetting dataset by time_indicies...")
            ds_subset = ds.isel(time=time_indices)

            if ds_subset.time.size > 0:
                print("Appending dataset...")
                datasets.append(ds_subset)

            ds.close()
            gc.collect()

        if len(datasets) > 0:
            # Concatenate datasets
            print(f"Concatenating {len(datasets)} datasets...")
            combined_ds = xr.concat(datasets, dim="time")

            combined_ds = attrs_manager.standardize_dataset_global_attrs(
                combined_ds, config, location, "a2", [str(f) for f in source_filenames]
            )

            temporal_string = time_manager.generate_temporal_attrs(combined_ds)[
                "standard_name"
            ]

            # Generate output filename
            data_level_file_name = (
                file_name_convention_manager.generate_filename_for_data_level(
                    combined_ds,
                    location["output_name"],
                    config["dataset"]["name"],
                    "a2",
                    temporal=temporal_string,
                )
            )

            output_path = Path(output_dir, f"{count:03d}.{data_level_file_name}")

            print(f"Saving partition file: {output_path}...")
            combined_ds.to_netcdf(output_path, encoding=config["dataset"]["encoding"])
            partition_files.append(output_path)
            print(f"Saved partition file: {output_path}!")

            # Cleanup
            combined_ds.close()
            for ds in datasets:
                ds.close()
            datasets.clear()
            gc.collect()

        datasets = []
        gc.collect()

    return partition_files


class AsyncFileWriter:
    """
    A helper class that manages asynchronous file writing using a worker pool.
    Optimized for HPC environments with scratch directories.
    Uses memory measurement to determine optimal worker count.
    """

    def __init__(self, max_workers=None, queue_size=100, sample_dataset=None):
        # Determine optimal number of workers based on available resources
        # If a sample dataset is provided, use it to measure memory requirements
        if max_workers is None:
            max_workers = self._determine_optimal_workers(sample_dataset)

        self.max_workers = max_workers
        print(f"AsyncFileWriter initialized with {self.max_workers} workers")

        # Create a queue to hold pending save operations
        self.save_queue = queue.Queue(maxsize=queue_size)

        # Keep track of all submitted tasks
        self.results = []
        self.futures = []

        # Start the thread pool
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        )

        # Start a monitoring thread to handle completed futures
        self.monitor_thread = threading.Thread(
            target=self._monitor_futures, daemon=True
        )
        self.monitor_thread.start()

        # Flag to signal shutdown
        self.shutdown_flag = False

    def _measure_memory_usage(self, dataset):
        """Measure memory usage for a sample dataset operation"""
        if psutil is None or dataset is None:
            return None

        try:
            # Record memory before operation
            process = psutil.Process(os.getpid())
            mem_before = process.memory_info().rss

            # Perform a similar operation to what we'll do in the worker
            # This creates a copy and measures the overhead
            dataset_copy = dataset.copy(deep=True)

            # Force computation to ensure memory is allocated
            dataset_copy.compute()

            # Measure memory after operation
            mem_after = process.memory_info().rss

            # Calculate difference in MB
            mem_used_mb = (mem_after - mem_before) / (1024 * 1024)

            # Add a safety margin (50%)
            mem_per_worker_mb = mem_used_mb * 1.5

            print(f"Memory measurement: operation used {mem_used_mb:.2f} MB")
            print(
                f"Estimated memory per worker with safety margin: {mem_per_worker_mb:.2f} MB"
            )

            return mem_per_worker_mb
        except Exception as e:
            print(f"Error measuring memory usage: {e}")
            return None

    def _determine_optimal_workers(self, sample_dataset=None):
        """Determine the optimal number of I/O workers based on measured memory usage"""
        try:
            if psutil is None:
                print("psutil not installed. Install with: pip install psutil")
                return 4

            # Get CPU count
            cpu_count = psutil.cpu_count(logical=False)  # Physical cores only
            if cpu_count is None:
                cpu_count = psutil.cpu_count(logical=True)  # Logical cores as fallback

            # Get available memory in MB
            mem_info = psutil.virtual_memory()
            avail_mem_mb = mem_info.available / (1024 * 1024)

            # For I/O bound tasks on HPC systems, we can use more workers than CPU cores
            io_optimal = min(cpu_count * 2, 32)  # Cap at 32 threads

            # Measure memory usage if sample dataset is provided
            mem_per_worker_mb = None
            if sample_dataset is not None:
                mem_per_worker_mb = self._measure_memory_usage(sample_dataset)

            # If we have a measured memory value, use it to calculate memory-based worker count
            if mem_per_worker_mb is not None and mem_per_worker_mb > 0:
                # Calculate how many workers we can run based on available memory
                # Use 80% of available memory to leave room for the main process
                mem_optimal = max(1, int((avail_mem_mb * 0.8) / mem_per_worker_mb))
                print(
                    f"Memory-based worker calculation: {avail_mem_mb:.1f}MB available / {mem_per_worker_mb:.1f}MB per worker = {mem_optimal} workers"
                )
            else:
                # If we couldn't measure, use a conservative estimate (1GB per worker)
                mem_optimal = max(1, int(avail_mem_mb / 1024))
                print(
                    f"No memory measurement available. Conservative estimate: {avail_mem_mb/1024:.1f}GB available / 1GB per worker = {mem_optimal} workers"
                )

            # Choose the minimum of the two constraints
            workers = min(io_optimal, mem_optimal)

            print(
                f"System resources: {cpu_count} CPU cores, {avail_mem_mb/1024:.1f}GB available memory"
            )
            print(
                f"Worker allocation: {workers} workers (I/O optimal: {io_optimal}, memory optimal: {mem_optimal})"
            )

            return workers
        except Exception as e:
            print(
                f"Error determining optimal worker count: {e}. Defaulting to 4 workers"
            )
            return 4

    def submit(self, ds, output_path, encoding):
        """Submit a dataset to be saved asynchronously"""
        # Create a deep copy to avoid race conditions with the main thread
        ds_copy = ds.copy(deep=True)

        # Submit the task to the executor
        future = self.executor.submit(self._save_file, ds_copy, output_path, encoding)
        self.futures.append((future, output_path))
        return output_path

    def _save_file(self, ds, output_path, encoding):
        """Worker function that saves a single file with HPC optimizations"""
        try:
            print(f"Saving file: {output_path}")

            # For HPC scratch directories, sometimes explicit sync/flush helps
            # Use smaller chunks for better parallel I/O performance
            ds.to_netcdf(
                output_path,
                encoding=encoding,
                compute=True,  # Ensure computation happens in this thread
                unlimited_dims=None,  # Avoid unlimited dimensions for better performance
            )

            # Ensure file is completely written
            os.fsync(os.open(str(output_path), os.O_RDONLY))

            print(f"Successfully saved: {output_path}")
            return output_path
        except Exception as e:
            print(f"Error saving {output_path}: {e}")
            raise

    def _monitor_futures(self):
        """Monitor thread that checks for completed futures and collects results"""
        while not self.shutdown_flag or self.futures:
            # Process any completed futures
            still_running = []
            for future, path in self.futures:
                if future.done():
                    try:
                        result = future.result()
                        self.results.append(result)
                    except Exception as e:
                        print(f"Error in file writing task for {path}: {e}")
                else:
                    still_running.append((future, path))

            # Update the list of running futures
            self.futures = still_running

            # Sleep briefly to avoid spinning
            time.sleep(0.1)

    def wait_all(self):
        """Wait for all pending save operations to complete"""
        print(f"Waiting for {len(self.futures)} file save operations to complete...")

        # Wait for all futures to complete
        for future, path in list(self.futures):
            try:
                result = future.result()
                if result not in self.results:
                    self.results.append(result)
            except Exception as e:
                print(f"Error in file writing task for {path}: {e}")

        # Clear the futures list
        self.futures = []
        return self.results

    def shutdown(self):
        """Shut down the writer and wait for all tasks to complete"""
        self.shutdown_flag = True
        results = self.wait_all()
        self.executor.shutdown()
        return results


def single_timestamp_partition(
    config, location_key, force_reprocess=False, max_workers=None
):
    """
    Partitions each input file into separate files with one timestamp per file.
    Uses non-blocking I/O optimized for HPC environments.
    Measures memory usage on first dataset to determine optimal worker count.

    Args:
        config: Configuration dictionary
        location_key: Key for the location in the config
        force_reprocess: Whether to force reprocessing if files already exist
        max_workers: Maximum number of parallel workers for file saving.
                     If None, automatically determines based on system.

    Returns:
        List of generated output file paths
    """
    location = config["location_specification"][location_key]
    output_dir = file_manager.get_standardized_partition_output_dir(config, location)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_directory = file_manager.get_standardized_output_dir(config, location)
    # Sort input files to ensure we process them in order
    sorted_input_files = sorted(list(input_directory.rglob("*.nc")))

    if not sorted_input_files:
        print("No input files found to process.")
        return []

    # Keep track of all timestamps to verify ordering at the end
    all_timestamps = []

    # First, process a sample timestamp to measure memory requirements
    sample_dataset = None
    try:
        # Open the first file to get a sample dataset
        print(
            f"Opening first file to measure memory requirements: {sorted_input_files[0]}"
        )
        first_ds = xr.open_dataset(sorted_input_files[0])

        # Get the first timestamp
        first_timestamp = pd.to_datetime(first_ds.time.values[0])

        # Create a sample dataset with a single timestamp
        time_index = 0  # Just use the first time index
        sample_dataset = first_ds.isel(time=[time_index])

        print(f"Created sample dataset with timestamp {first_timestamp}")

        # Close the dataset
        first_ds.close()
    except Exception as e:
        print(f"Error creating sample dataset: {e}")

    # Set up the async file writer with the sample dataset for memory measurement
    file_writer = AsyncFileWriter(
        max_workers=max_workers, sample_dataset=sample_dataset
    )

    # Global counter for output files
    file_counter = 1
    timestamp_counter = 1

    try:
        for input_file in sorted_input_files:
            print(f"Processing file: {input_file}")

            # Open the dataset
            print(f"Reading: {input_file}")
            ds = xr.open_dataset(input_file)

            # Get all timestamps in this file
            timestamps = pd.to_datetime(ds.time.values)

            # Process each timestamp individually
            for timestamp in timestamps:
                print(f"Processing timestamp {timestamp_counter}: {timestamp}")

                # Record this timestamp
                all_timestamps.append(timestamp)

                # Subset to just this timestamp
                time_index = np.where(ds.time.values == np.datetime64(timestamp))[0][0]
                print(f"Selecting time index: {time_index}")
                ds_single_time = ds.isel(time=[time_index])

                # Format timestamp for filename
                timestamp_str = pd.to_datetime(timestamp).strftime("%Y%m%dT%H%M%S")

                print("Building output_file name...")
                # Generate output filename
                data_level_file_name = (
                    file_name_convention_manager.generate_filename_for_data_level(
                        ds_single_time,
                        location["output_name"],
                        config["dataset"]["name"],
                        "a2",
                    )
                )

                output_path = Path(
                    output_dir,
                    f"{file_counter:03d}_{timestamp_str}.{data_level_file_name}",
                )

                # Submit the dataset for async saving
                file_writer.submit(
                    ds_single_time, output_path, config["dataset"]["encoding"]
                )

                # Increment file counter for next file
                file_counter += 1
                timestamp_counter += 1

            # Close the dataset
            ds.close()
            gc.collect()

        # Verify timestamps are in order
        if all_timestamps:
            for i in range(1, len(all_timestamps)):
                if all_timestamps[i] < all_timestamps[i - 1]:
                    error_msg = f"Timestamps not in order: {all_timestamps[i-1]} followed by {all_timestamps[i]}"
                    print(error_msg)
                    raise ValueError(error_msg)

        # Wait for all file saving operations to complete
        partition_files = file_writer.wait_all()

        print(
            f"Successfully processed {len(partition_files)} timestamps across {len(sorted_input_files)} files"
        )
        return partition_files

    finally:
        # Ensure proper cleanup of resources
        file_writer.shutdown()
