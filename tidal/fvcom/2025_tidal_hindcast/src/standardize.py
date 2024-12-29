# Create one giant dataset using the valid_timestamps

from . import time_manager


def standardize_dataset(config, location, nc_files, valid_timestamps_df):
    # At this step the timestamps can have duplicates. This step uses the strategy defined
    # in the config to remove duplicate timestamps
    drop_duplicates_keep_strategy = config["time_specification"][
        "drop_duplicate_timestamps_keep_strategy"
    ]
    time_df = valid_timestamps_df.drop_duplicates(keep=drop_duplicates_keep_strategy)

    time_manager.does_time_match_specification(
        time_df["Timestamp"], location["expected_delta_t_seconds"]
    )

    for _, this_df in time_df.group_by(["source_file"]):
        print(len(this_df))
        print(this_df["Timestamp"].iloc[0])
        print(this_df["Timestamp"].iloc[-1])
