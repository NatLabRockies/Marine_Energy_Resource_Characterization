from config import config
from src.cli import parse_args
from src.file_manager import get_specified_nc_files

from src.verify import verify_dataset
from src.standardize import standardize_dataset
from src.partition_by_time import partition_by_time
from src.derive_vap_fvcom import derive_vap
from src.calculate_vap_average import calculate_vap_average


if __name__ == "__main__":
    args = parse_args(config)

    # Access the location config
    location = config["location_specification"][args.location]
    output_type = args.output_type

    print(f"Standardizing {location} tidal dataset for output type {output_type}....")

    print("Finding nc files...")
    try:
        nc_files = get_specified_nc_files(config, location)
        print(f"Found {len(nc_files)} files!")
    except PermissionError:
        print("Permissions error accessing original files")
        nc_files = []

    print("Step 1: Verifying Dataset Integrity...")
    valid_timestamps_df = verify_dataset(config, location, nc_files)

    print("Step 2: Modifying Original Dataset to Create a Standardized Dataset...")
    valid_std_files_df = standardize_dataset(config, args.location, valid_timestamps_df)

    print("Step 3: Partitioning Standardized Dataset by Time...")
    partition_by_time(config, args.location, valid_std_files_df)

    print("Step 4: Calculating Derived Value Added Products...")
    derive_vap(config, args.location)

    print("Step 5: Calculating Yearly Averages...")
    calculate_vap_average(config, args.location)
