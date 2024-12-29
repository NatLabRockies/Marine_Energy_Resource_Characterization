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


def get_output_dirs(config):
    output_dirs = config["dir"]["output"]
    return {
        "standardized": Path(output_dirs["standardized"]),
        "vap": Path(output_dirs["vap"]),
        "summary_vap": Path(output_dirs["summary_vap"]),
    }


def get_standardized_output_dir(config):
    return get_output_dirs(config)["standardized"]


def get_vap_output_dir(config):
    return get_output_dirs(config)["vap"]


def get_summary_vap_output_dir(config):
    return get_output_dirs(config)["summary_vap"]
