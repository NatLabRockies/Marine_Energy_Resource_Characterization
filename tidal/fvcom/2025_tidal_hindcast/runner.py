from config import config
from src.cli import parse_args
from src.file_manager import get_specified_nc_files

from src.verify import verify_dataset
from src.standardize import standardize_dataset


if __name__ == "__main__":
    args = parse_args(config)

    # Access the location config
    location = config["location_specification"][args.location]
    output_type = args.output_type

    print(f"Standardizing {location} tidal dataset for output type {output_type}....")

    print("Finding nc files...")
    nc_files = get_specified_nc_files(config, location)
    print(f"Found {len(nc_files)} files!")

    print("Step 1: Verifying Dataset Integrity...")
    valid_timestamps_df = verify_dataset(config, location, nc_files)

    print("Step 2: Modifying Original Dataset to Create a Standardized Dataset...")
    standardize_dataset(config, args.location, valid_timestamps_df)

    print("Step 3: Partitioning Standardized Dataset by Time...")
