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


def get_output_dirs(config, location, use_temp_base_path=False, omit_base_path=False):
    output_location_name = location["output_name"]
    version = f"v{config['dataset']['version']}"
    base_path = Path(config["dir"]["base"]).resolve()
    output_dirs = config["dir"]["output"]

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

    def build_path(str_path_with_vars):
        str_path = str_path_with_vars.replace(
            "<location>", output_location_name
        ).replace("<version>", version)

        if omit_base_path:
            return Path(str_path)
        else:
            return Path(base_path, str_path)

    paths = {}
    for key, value in output_dirs.items():
        paths[key] = build_path(value)

    # Only create directories if we're using full paths
    if not omit_base_path:
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


def get_vap_daily_compressed_output_dir(config, location):
    return get_output_dirs(config, location)["vap_daily_compressed"]


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


def get_manifest_output_dir(config, manifest_version=None):
    """
    Get manifest output directory for the given or latest manifest version.

    The manifest directory is global (not location-specific) and follows the pattern:
    {base_path}/manifest/v{manifest_version}/

    Args:
        config: Configuration dictionary
        manifest_version: Optional specific manifest version. If None, uses config version.

    Returns:
        Path to the manifest output directory
    """
    base_path = Path(config["dir"]["base"]).resolve()

    if manifest_version is None:
        manifest_version = config["manifest"]["version"]

    manifest_path_template = config["dir"]["output"]["manifest"]
    manifest_path = manifest_path_template.replace(
        "<manifest_version>", manifest_version
    )

    return base_path / manifest_path


def find_latest_manifest_version(config):
    """
    Scan filesystem to find the latest manifest version by semver.

    Looks for directories matching pattern: {base_path}/manifest/v*

    Args:
        config: Configuration dictionary

    Returns:
        Tuple of (latest_version_string, latest_version_path) or (None, None) if none found
    """
    base_path = Path(config["dir"]["base"]).resolve()
    manifest_base = base_path / "manifest"

    if not manifest_base.exists():
        return None, None

    # Find all version directories
    version_dirs = sorted(manifest_base.glob("v*"))

    if not version_dirs:
        return None, None

    # Parse versions and find highest
    def parse_semver(version_str):
        """Parse 'v1.2.3' or '1.2.3' into tuple (1, 2, 3) for comparison"""
        clean = version_str.lstrip("v")
        try:
            parts = clean.split(".")
            return tuple(int(p) for p in parts)
        except (ValueError, AttributeError):
            return (0, 0, 0)

    latest_dir = max(version_dirs, key=lambda d: parse_semver(d.name))
    latest_version = latest_dir.name.lstrip("v")

    return latest_version, latest_dir


def validate_manifest_version(config):
    """
    Validate that config manifest version matches the latest on filesystem.

    Args:
        config: Configuration dictionary

    Returns:
        Tuple of (is_valid, config_version, filesystem_version, manifest_dir)

    Raises:
        ValueError: If versions don't match
    """
    config_version = config["manifest"]["version"]
    fs_version, fs_path = find_latest_manifest_version(config)

    if fs_version is None:
        raise ValueError(
            f"No manifest directories found on filesystem. "
            f"Config specifies version {config_version}. "
            f"Expected directory at: {get_manifest_output_dir(config)}"
        )

    if config_version != fs_version:
        raise ValueError(
            f"Manifest version mismatch! "
            f"Config specifies version '{config_version}' but filesystem has "
            f"latest version '{fs_version}' at {fs_path}. "
            f"Please update config['manifest']['version'] to match."
        )

    return True, config_version, fs_version, fs_path
