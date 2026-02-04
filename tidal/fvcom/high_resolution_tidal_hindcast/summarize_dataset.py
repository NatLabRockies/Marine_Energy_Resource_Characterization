from config import config
from src.cli import parse_partition_args

from src.calculate_vap_average import (
    calculate_vap_yearly_average,
    calculate_vap_monthly_average,
)


if __name__ == "__main__":
    args = parse_partition_args(config)

    # Access the location config
    location = config["location_specification"][args.location]

    batch_size = args.batch_size
    batch_number = args.batch_num

    print(
        f"Create VAP Summary Dataset for {args.location} for batch {batch_number} of size {batch_size}..."
    )
    calculate_vap_yearly_average(
        config, args.location, batch_size=batch_size, batch_number=batch_number
    )

    # calculate_vap_monthly_average(
    #     config, args.location, batch_size=batch_size, batch_number=batch_number
    # )
