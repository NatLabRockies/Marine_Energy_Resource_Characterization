from config import config
from src.cli import parse_partition_args
from src.file_manager import get_specified_nc_files

from src.vap_simple_create_parquet_all_time_partition import (
    partition_vap_into_parquet_dataset,
)


if __name__ == "__main__":
    args = parse_partition_args(config)

    # Access the location config
    location = config["location_specification"][args.location]

    batch_size = args.batch_size
    batch_number = args.batch_num

    print(f"Create Parquet Partition Dataset for {args.location}...")
    partition_vap_into_parquet_dataset(
        config,
        args.location,
        batch_size=batch_size,
        batch_number=batch_number,
        skip_existing=args.skip_existing,
    )
