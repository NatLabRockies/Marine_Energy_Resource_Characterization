import os

from pathlib import Path


def get_specified_nc_files(config, location):
    base_path = Path(config["dir"]["base"]).resolve()
    original_data_dir = Path(base_path, config["dir"]["input"]["original"]).resolve()
    original_data_dir = str(original_data_dir).replace(
        "<location>", location["output_name"]
    )
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


def get_output_dirs(config, location, use_temp_base_path=False):
    output_location_name = location["output_name"]
    base_path = Path(config["dir"]["base"]).resolve()

    if use_temp_base_path:
        # https://nrel.github.io/HPC/Documentation/Systems/Kestrel/Running/example_sbatch/
        kestrel_tmp_dir = os.getenv("TMPDIR")
        temp_dir = Path(kestrel_tmp_dir).resolve()
        print(f"Temporary directory resolved to: {temp_dir}...")

        if temp_dir.exists() is False:
            print(
                f"Temporary directory {temp_dir} does not exist! "
                "Please check your environment variables."
                "Using default base path instead."
            )
        else:
            base_path = temp_dir

    output_dirs = config["dir"]["output"]
    paths = {
        "tracking": Path(
            base_path,
            output_dirs["tracking"].replace("<location>", output_location_name),
        ),
        "standardized": Path(
            base_path,
            output_dirs["standardized"].replace("<location>", output_location_name),
        ),
        "standardized_partition": Path(
            base_path,
            output_dirs["standardized_partition"].replace(
                "<location>", output_location_name
            ),
        ),
        "vap": Path(
            base_path, output_dirs["vap"].replace("<location>", output_location_name)
        ),
        "monthly_summary_vap": Path(
            base_path,
            output_dirs["monthly_summary_vap"].replace(
                "<location>", output_location_name
            ),
        ),
        "yearly_summary_vap": Path(
            base_path,
            output_dirs["yearly_summary_vap"].replace(
                "<location>", output_location_name
            ),
        ),
        "yearly_summary_vap_by_face": Path(
            base_path,
            output_dirs["yearly_summary_vap"].replace(
                "<location>", output_location_name
            ),
            "by_face",
        ),
        "vap_partition": Path(
            base_path,
            output_dirs["vap_partition"].replace("<location>", output_location_name),
        ),
        "vap_summary_parquet": Path(
            base_path,
            output_dirs["vap_summary_parquet"].replace(
                "<location>", output_location_name
            ),
        ),
        "vap_atlas_summary_parquet": Path(
            base_path,
            output_dirs["vap_atlas_summary_parquet"].replace(
                "<location>", output_location_name
            ),
        ),
        "combined_vap_atlas": Path(base_path, output_dirs["combined_vap_atlas"]),
        "hsds": Path(
            base_path,
            output_dirs["hsds"].replace("<location>", output_location_name),
        ),
        "hsds_temp": Path(
            base_path,
            output_dirs["hsds_temp"].replace("<location>", output_location_name),
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


def get_monthly_summary_vap_output_dir(config, location):
    return get_output_dirs(config, location)["monthly_summary_vap"]


def get_yearly_summary_vap_output_dir(config, location):
    return get_output_dirs(config, location)["yearly_summary_vap"]


def get_yearly_summary_by_face_vap_output_dir(config, location):
    return get_output_dirs(config, location)["yearly_summary_vap_by_face"]


def get_vap_partition_output_dir(config, location, use_temp_base_path=False):
    return get_output_dirs(config, location, use_temp_base_path)["vap_partition"]


def get_vap_summary_parquet_dir(config, location):
    return get_output_dirs(config, location)["vap_summary_parquet"]


def get_vap_atlas_summary_parquet_dir(config, location):
    return get_output_dirs(config, location)["vap_atlas_summary_parquet"]


def get_combined_vap_atlas(config, location):
    return get_output_dirs(config, location)["combined_vap_atlas"]


def get_hsds_output_dir(config, location):
    """Get HSDS output directory for the location"""
    return get_output_dirs(config, location)["hsds"]


def get_hsds_temp_dir(config, location):
    """Get HSDS temporary directory for individual chunk files"""
    return get_output_dirs(config, location)["hsds_temp"]


def get_hsds_final_file_path(config, location):
    """Get final HSDS file path with versioned naming"""
    hsds_dir = get_hsds_output_dir(config, location)
    output_name = location["output_name"]
    dataset_name = config["dataset"]["name"]
    version = config["dataset"]["version"]

    filename = f"{output_name}.{dataset_name}.hsds.v{version}.h5"
    return hsds_dir / filename
