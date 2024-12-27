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
        result.append(nc_files)

    return result
