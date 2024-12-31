from pathlib import Path


def get_specified_nc_files(config, location):
    original_data_dir = Path(config["dir"]["input"]["original"]).resolve()
    search_dir = Path(original_data_dir, location["base_dir"]).resolve()
    if search_dir.exists() is not True:
        raise ValueError(f"Directory Error: {search_dir} does not exist!")

    result = []
    paths = location["files"]
    for path in paths:
        nc_files = sorted(list(search_dir.rglob(path)))
        result.extend(nc_files)

    if "files_to_exclude" in location:
        result = [f for f in result if f.name not in location["files_to_exclude"]]

    return result


def get_output_dirs(config, location):
    output_location_name = location["output_name"]
    output_dirs = config["dir"]["output"]
    paths = {
        "tracking": Path(
            output_dirs["tracking"].replace("<location>", output_location_name)
        ),
        "standardized": Path(
            output_dirs["standardized"].replace("<location>", output_location_name)
        ),
        "standardized_partition": Path(
            output_dirs["standardized_partition"].replace(
                "<location>", output_location_name
            )
        ),
        "vap": Path(output_dirs["vap"].replace("<location>", output_location_name)),
        "summary_vap": Path(
            output_dirs["summary_vap"].replace("<location>", output_location_name)
        ),
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)

    return paths


def get_tracking_output_dir(config, location):
    return get_output_dirs(config, location)["tracking"]


def get_standardized_output_dir(config, location):
    return get_output_dirs(config, location)["standardized"]


def get_standardized_partition_output_dir(config, location):
    return get_output_dirs(config, location)["standardized_partition"]


def get_vap_output_dir(config, location):
    return get_output_dirs(config, location)["vap"]


def get_summary_vap_output_dir(config, location):
    return get_output_dirs(config, location)["summary_vap"]
