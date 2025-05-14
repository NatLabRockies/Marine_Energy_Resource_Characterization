from config import config

from src.vap_simple_create_parquet_all_time_partition import (
    partition_vap_into_parquet_dataset,
)


if __name__ == "__main__":
    partition_vap_into_parquet_dataset(
        config, "cook_inlet", batch_size=2, batch_number=0
    )
