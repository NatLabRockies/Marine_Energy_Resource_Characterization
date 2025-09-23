from pathlib import Path

import pandas as pd

from config import config
from src.cli import parse_args
from src.file_manager import get_specified_nc_files, get_tracking_output_dir

from src.verify import verify_dataset
from src.standardize import standardize_dataset
from src.partition_by_time import partition_by_time
from src.derive_vap_fvcom import (
    calculate_and_save_face_center_precalculations,
    derive_vap,
)
from src.calculate_vap_average import (
    calculate_vap_monthly_average,
    calculate_vap_yearly_average,
)

# from src.vap_create_parquet_all_time_partition import partition_vap_into_parquet_dataset
# from src.vap_hybrid_create_parquet_all_time_partition import (
#     partition_vap_into_parquet_dataset,
# )

# from src.vap_h5_create_parquet_all_time_partition import (
#     partition_vap_into_parquet_dataset,
# )

# from src.vap_simple_create_parquet_all_time_partition import (
#     partition_vap_into_parquet_dataset,
# )

# from src.vap_optimized_create_all_time_partition import (
#     partition_vap_into_parquet_dataset,
# )
from src.vap_create_parquet_summary import convert_nc_summary_to_parquet


if __name__ == "__main__":
    args = parse_args(config)

    # Access the location config
    location = config["location_specification"][args.location]
    output_type = args.output_type

    # print(f"Standardizing {location} tidal dataset for output type {output_type}....")

    # print("Finding nc files...")
    # try:
    #     nc_files = get_specified_nc_files(config, location)
    #     print(f"Found {len(nc_files)} files!")
    # except PermissionError:
    #     print("Permissions error accessing original files")
    #     nc_files = []
    #
    # print("Step 1: Verifying Dataset Integrity...")
    # valid_timestamps_df = verify_dataset(config, location, nc_files)

    # tracking_folder = get_tracking_output_dir(config, location)
    # tracking_path = Path(
    #     tracking_folder, f"{location['output_name']}_verify_step_tracking.parquet"
    # )
    # valid_timestamps_df = pd.read_parquet(tracking_path)
    #
    # print("Step 2: Modifying Original Dataset to Create a Standardized Dataset...")
    # valid_std_files_df = standardize_dataset(
    #     config, args.location, valid_timestamps_df, skip_if_verified=False
    # )
    #
    # print("Step 3: Partitioning Standardized Dataset by Time...")
    # partition_by_time(config, args.location, valid_std_files_df)

    # print("Step 4: Calculating Derived Value Added Products...")
    # face_precalculations_path = calculate_and_save_face_center_precalculations(
    #     config,
    #     args.location,
    #     skip_if_precalculated=True,
    # )
    #
    # derive_vap(
    #     config,
    #     args.location,
    #     skip_if_output_files_exist=False,
    # )
    #
    # The following doesn't work anymore due to memory issues
    # Summary computations must use dispatch_summarize_jobs.py
    #
    # print("Step 5: Calculating Monthly Averages...")
    # calculate_vap_monthly_average(config, args.location)
    #
    # print("Step 6: Calculating Yearly Average...")
    # calculate_vap_yearly_average(config, args.location)

    # print("Step 7: Create Parquet Partition Dataset...")
    # partition_vap_into_parquet_dataset(config, args.location)

    print("Step 8: Create Summary Parquet Dataset...")
    convert_nc_summary_to_parquet(config, args.location)
