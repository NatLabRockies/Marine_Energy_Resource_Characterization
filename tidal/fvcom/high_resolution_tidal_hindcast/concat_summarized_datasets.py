from config import config
from src.cli import parse_partition_args

from src.calculate_vap_average import (
    combine_monthly_face_files,
    combine_yearly_face_files,
)


if __name__ == "__main__":
    args = parse_partition_args(config)

    print(args)

    print(dir(args))

    # Access the location config
    location = config["location_specification"][args.location]

    batch_number = args.batch_num

    if batch_number == 0:
        print("Combining monthly face files")
        combine_monthly_face_files(config, location)
    else:
        print("Combining yearly face files")
        combine_yearly_face_files(config, location)
