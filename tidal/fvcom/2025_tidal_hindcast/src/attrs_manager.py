import getpass
import json
import os
import platform
import socket
import subprocess

from datetime import datetime, timezone
from urllib.parse import urlparse

import numpy as np


def get_system_info():
    """Get detailed system information."""
    try:
        return {
            "hostname": socket.gethostname(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
        }

    except Exception as e:
        print(f"Error getting system info: {e}")
        return None


def get_history_string():
    """Generate a detailed history string with all system information."""
    try:
        username = getpass.getuser()
        timestamp = datetime.now().isoformat()
        sys_info = get_system_info()

        if sys_info:
            system_details = (
                f"on {sys_info['hostname']} "
                f"(OS: {sys_info.get('os_name', sys_info['system'])}, "
                f"Kernel: {sys_info['release']}, "
                f"Architecture: {sys_info['machine']}, "
                f"Processor: {sys_info['processor']}, "
                f"Python: {sys_info['python_version']})"
            )
        else:
            system_details = ""

        return f"Ran by {username} {system_details} at {timestamp}"
    except Exception as e:
        print(f"Error generating history string: {e}")
        return None


def run_git_command(command):
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
            shell=True,  # Note: Be careful with shell=True in production code
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Git command failed: {e}")
        return None


def extract_git_repo_versioning():
    try:
        # Get the current commit SHA
        sha = run_git_command("git rev-parse HEAD")
        if not sha:
            raise ValueError("Could not get commit SHA")

        # Get the remote URL
        origin_url = run_git_command("git config --get remote.origin.url")
        if not origin_url:
            raise ValueError("Could not get remote origin URL")

        # Handle different URL formats
        if origin_url.startswith("git@"):
            # Convert SSH format to HTTPS
            # Handles both github.com and custom domains like github.nrel.gov
            ssh_parts = origin_url.split("@")[1].split(":")
            domain = ssh_parts[0]
            path = ssh_parts[1]
            if path.endswith(".git"):
                path = path[:-4]
            origin_url = f"https://{domain}/{path}"
        elif origin_url.startswith("https://"):
            # Already in HTTPS format, just remove .git if present
            if origin_url.endswith(".git"):
                origin_url = origin_url[:-4]

        # Create the full URL with commit SHA
        code_url = f"{origin_url}/tree/{sha}"

        # Validate the URL structure
        try:
            parsed_url = urlparse(code_url)
            if not all([parsed_url.scheme, parsed_url.netloc, parsed_url.path]):
                raise ValueError("Invalid URL structure")
        except Exception:
            raise ValueError(f"Invalid URL generated: {code_url}")

        return {"code_url": code_url, "code_version": sha}
    except Exception as e:
        print(f"Error getting git info: {e}")
        return None


def compute_geospatial_bounds(ds, crs_string=None):
    """
    Compute geospatial bounds and related quantities from an xarray dataset
    with an unstructured grid.

    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset containing latitude and longitude coordinates.
        Must have units specified for both coordinates.
    crs_string : str, optional
        The coordinate reference system string. If None, will attempt to
        read from ds.attrs['geospatial_bounds_crs']

    Returns
    -------
    dict
        Dictionary containing computed geospatial quantities following ACDD conventions.
        For unstructured grids, resolution fields are left as None since there is no
        uniform targeted spacing.

    Raises
    ------
    ValueError
        If required units are missing from coordinates
        If CRS string is not available in either input parameter or dataset attributes
    """
    # Initialize output dictionary
    bounds = {
        "geospatial_bounds": None,
        "geospatial_bounds_crs": None,
        "geospatial_lat_max": None,
        "geospatial_lat_min": None,
        "geospatial_lat_units": None,
        "geospatial_lat_resolution": None,  # Will remain None for unstructured grid
        "geospatial_lon_max": None,
        "geospatial_lon_min": None,
        "geospatial_lon_units": None,
        "geospatial_lon_resolution": None,  # Will remain None for unstructured grid
    }

    # Get latitude and longitude variables
    lat = ds.lat_node
    lon = ds.lon_node

    # Check for required units
    if not hasattr(lat, "units"):
        raise ValueError("Latitude coordinate must have units specified")
    if not hasattr(lon, "units"):
        raise ValueError("Longitude coordinate must have units specified")

    # Set units
    bounds["geospatial_lat_units"] = lat.units
    bounds["geospatial_lon_units"] = lon.units

    # Compute simple min/max bounds
    bounds["geospatial_lat_max"] = float(lat.max().values)
    bounds["geospatial_lat_min"] = float(lat.min().values)
    bounds["geospatial_lon_max"] = float(lon.max().values)
    bounds["geospatial_lon_min"] = float(lon.min().values)

    # Get CRS from input parameter or dataset attributes
    if crs_string is not None:
        bounds["geospatial_bounds_crs"] = crs_string
    elif "geospatial_bounds_crs" in ds.attrs:
        bounds["geospatial_bounds_crs"] = ds.attrs["geospatial_bounds_crs"]
    else:
        raise ValueError(
            "CRS string must be provided either as input parameter or in dataset attributes"
        )

    # Simple bounding box representation
    bounds["geospatial_bounds"] = (
        f"POLYGON (({bounds['geospatial_lon_min']} {bounds['geospatial_lat_min']}, "
        f"{bounds['geospatial_lon_max']} {bounds['geospatial_lat_min']}, "
        f"{bounds['geospatial_lon_max']} {bounds['geospatial_lat_max']}, "
        f"{bounds['geospatial_lon_min']} {bounds['geospatial_lat_max']}, "
        f"{bounds['geospatial_lon_min']} {bounds['geospatial_lat_min']}))"
    )

    return bounds


def compute_modification_dates(ds):
    """
    Get date-related attributes for an xarray Dataset.
    Uses 'history' or 'date_created' field for date_created,
    and current timestamp for modifications.

    Parameters:
    ds (xarray.Dataset): Input dataset to extract dates from

    Returns:
    dict: Dictionary of attribute names and their values

    Raises:
    ValueError: If neither 'history' nor 'date_created' attributes exist
    """
    result = {}

    # Try to get date_created from either history or existing date_created
    if "date_created" in ds.attrs:
        result["date_created"] = ds.attrs["date_created"]
    elif "history" in ds.attrs:
        history = ds.attrs["history"]
        if "model started at:" in history:
            date_str = history.split("model started at:")[1].strip()
            try:
                created_date = datetime.strptime(date_str, "%m/%d/%Y   %H:%M")
                result["date_created"] = created_date.isoformat()
            except ValueError:
                result["date_created"] = None
        else:
            raise ValueError(
                f"Unexpected history attr. Expecting 'model started at' got {history}"
            )
    else:
        raise ValueError(
            "Neither 'history' nor 'date_created' attributes found in dataset"
        )

    # Get current time in UTC
    now = datetime.now(timezone.utc)
    current_time_str = now.isoformat()

    # Add remaining attributes
    result["date_modified"] = current_time_str
    result["date_metadata_modified"] = current_time_str

    return result


def compute_vertical_attributes(ds):
    """
    Compute vertical attributes for an FVCOM dataset with sigma layers.

    Parameters:
    ds (xarray.Dataset): Input dataset containing bathymetry and sigma layer variables

    Returns:
    dict: Dictionary of vertical attribute names and their values
    """
    vertical_attrs = {}

    # Get bathymetry values from both h and h_center
    h_values = []
    h_var = None
    if "h" in ds:
        h_values.extend(ds.h.values)
        h_var = ds.h
    if "h_center" in ds:
        h_values.extend(ds.h_center.values)
        if h_var is None:
            h_var = ds.h_center

    # Get units and positive direction from bathymetry attributes if available
    if h_var is not None:
        vertical_attrs["geospatial_vertical_units"] = h_var.attrs.get("units", None)
        vertical_attrs["geospatial_vertical_positive"] = h_var.attrs.get(
            "positive", None
        )
    else:
        vertical_attrs["geospatial_vertical_units"] = None
        vertical_attrs["geospatial_vertical_positive"] = None

    # Get maximum water level from zeta if available
    max_elevation = 0  # Default to 0 if no zeta
    if "zeta" in ds:
        max_elevation = float(np.nanmax(ds.zeta.values))

    if h_values:
        # Maximum depth is the deepest bathymetry point
        max_depth = float(np.nanmax(h_values))
        # Minimum is surface (including maximum water elevation)
        min_depth = -max_elevation  # Negative because depth is positive down

        vertical_attrs["geospatial_vertical_min"] = min_depth
        vertical_attrs["geospatial_vertical_max"] = max_depth
    else:
        vertical_attrs["geospatial_vertical_min"] = None
        vertical_attrs["geospatial_vertical_max"] = None

    # Compute vertical resolution if we have sigma layer information
    if "siglay_center" in ds:
        # Get number of sigma layers
        n_layers = ds.siglay_center.shape[0]

        vertical_attrs["geospatial_vertical_resolution"] = (
            f"Variable ({n_layers} sigma layers)"
        )
    else:
        vertical_attrs["geospatial_vertical_resolution"] = None

    # Set CRS based on sigma coordinates
    if "siglay_center" in ds and "standard_name" in ds.siglay_center.attrs:
        vertical_attrs["geospatial_bounds_vertical_crs"] = ds.siglay_center.attrs[
            "standard_name"
        ]
    else:
        vertical_attrs["geospatial_bounds_vertical_crs"] = None

    # Set vertical origin based on zeta standard_name if available
    if "zeta" in ds and "standard_name" in ds.zeta.attrs:
        if ds.zeta.attrs["standard_name"] == "sea_surface_height_above_geoid":
            vertical_attrs["geospatial_vertical_origin"] = "geoid"
        else:
            vertical_attrs["geospatial_vertical_origin"] = None
    else:
        vertical_attrs["geospatial_vertical_origin"] = None

    return vertical_attrs


def validate_attribute_changes(ds, new_attrs, allowed_changing_keys):
    """
    Validate that only allowed global attributes are being changed in the dataset.

    Parameters:
    -----------
    ds : xarray.Dataset
        The original dataset with current attributes
    new_attrs : dict
        Dictionary of new attributes to be added/updated
    allowed_changing_keys : list
        List of attribute names that are allowed to change

    Returns:
    --------
    bool
        True if all changes are valid

    Raises:
    -------
    ValueError
        If attributes that are not in allowed_changing_keys are being modified
    KeyError
        If new_attrs contains keys that don't exist and aren't in allowed_changing_keys
    """
    current_attrs = ds.attrs

    # Find keys that are changing
    changing_keys = []
    for key, new_value in new_attrs.items():
        # If key exists and value is different, or key doesn't exist
        if key in current_attrs:
            if current_attrs[key] != new_value:
                changing_keys.append(key)
        else:
            # New key being added
            changing_keys.append(key)

    # Find unauthorized changes
    unauthorized_changes = set(changing_keys) - set(allowed_changing_keys)

    if unauthorized_changes:
        error_msg = (
            f"Attempting to modify restricted attributes: {sorted(unauthorized_changes)}\n"
            f"Only these attributes can be modified: {sorted(allowed_changing_keys)}"
        )
        raise ValueError(error_msg)

    return True


def compute_file_hash(filepath):
    """Compute SHA-256 hash of a file in chunks to handle large files efficiently.

    Args:
        filepath (str): Path to the file to hash

    Returns:
        str: Hexadecimal string of the SHA-256 hash
    """
    import hashlib

    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        # Read in 64kb chunks for memory efficiency with large files
        for byte_block in iter(lambda: f.read(65536), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def create_file_metadata(path):
    """Create metadata dictionary for a single file including hash.

    Args:
        path (str): Path to the file

    Returns:
        dict: Dictionary containing filename, path and hash information
    """
    return {
        "filename": os.path.basename(path),
        "path": path,
        "hash": {"algorithm": "SHA-256", "value": compute_file_hash(path)},
    }


def process_input_files(input_files):
    return [create_file_metadata(path) for path in input_files]


def create_input_history_dict(
    ds, data_level, input_files, input_ds_is_original_model_output=False
):
    """Create or update the source files dictionary with file metadata.

    Args:
        ds (xarray.Dataset): The dataset containing existing attributes
        data_level (str): The data level identifier
        input_files (list): List of input file paths
        input_ds_is_original_model_output (bool): Flag indicating if this is original model output

    Returns:
        str: JSON string containing the source files dictionary
    """
    processed_files = process_input_files(input_files)

    if input_ds_is_original_model_output:
        input_history_dict = {
            data_level: processed_files,
        }
    else:
        input_history_dict = json.loads(ds.attrs["input_history_json"])
        input_history_dict[data_level] = processed_files

    return json.dumps(input_history_dict, indent=2, ensure_ascii=False)


def standardize_dataset_global_attrs(
    ds,
    config,
    location,
    data_level,
    input_files,
    input_ds_is_original_model_output=False,
    coordinate_reference_system_string=None,
):
    code_metadata = extract_git_repo_versioning()

    config_global_attrs = config["global_attributes"]

    geospatial_bounds = compute_geospatial_bounds(
        ds, coordinate_reference_system_string
    )

    modification_dates = compute_modification_dates(ds)

    vertical_attributes = compute_vertical_attributes(ds)

    # Use json to encode source files as a dict of lists of source files to track provenance
    input_history_json = create_input_history_dict(
        ds, data_level, input_files, input_ds_is_original_model_output
    )

    # Global attributes that will be recorded in the output dataset in `attrs` (attributes). These metadata are
    # used to record data provenance information (e.g., location, institution, etc),
    # construct datastream and file names (i.e., location_id, dataset_name, qualifier,
    # temporal, and data_level attributes), as well as provide metadata that is useful for
    # data users (e.g., title, description, ... ).
    standardized_attributes = {
        # Source: ACDD
        # A comma-separated list of the conventions that are followed by the dataset.
        # For files that follow this version of ACDD, include the string 'ACDD-1.3'.
        "Conventions": config_global_attrs["Conventions"],
        # Source: ACDD
        # A place to acknowledge various types of support for the project that produced this data.
        "acknowledgement": config_global_attrs["acknowledgement"],
        # Source: Global
        # Where the code is hosted.
        "code_url": code_metadata["code_url"],
        # Source: Global
        # Attribute that will be recorded automatically by the pipeline. A warning will be raised
        # if this is set in the config file. The code_version attribute reads the 'CODE_VERSION'
        # environment variable or parses the git history to determine the version of the code.
        "code_version": config["code"]["version"],
        # Source: IOOS
        # Country of the person or organization that operates a platform or network,
        # which collected the observation data.
        "creator_country": config_global_attrs["creator_country"],
        # Source: ACDD, IOOS
        # The email address of the person (or other creator type specified by the creator_type
        # attribute) principally responsible for creating this data.
        "creator_email": config_global_attrs["creator_email"],
        # Source: ACDD, IOOS
        # The institution of the creator; should uniquely identify the creator's institution.
        # This attribute's value should be specified even if it matches the value of
        # publisher_institution, or if creator_type is institution.
        "creator_institution": config_global_attrs["creator_institution"],
        # Source: IOOS
        # URL for the institution that collected the data. For clarity, it is recommended
        # that this field is specified even if the creator_type is institution and a
        # creator_url is provided.
        "creator_institution_url": config_global_attrs["creator_institution_url"],
        # Source: ACDD
        # The name of the person (or other creator type specified by the creator_type
        # attribute) principally responsible for creating this data.
        "creator_name": config_global_attrs["creator_name"],
        # Source: IOOS
        # IOOS classifier (https://mmisw.org/ont/ioos/sector) that best describes
        # the platform (network) operator's societal sector.
        "creator_sector": config_global_attrs["creator_sector"],
        # Source: IOOS
        # State or province of the person or organization that collected the data.
        "creator_state": config_global_attrs["creator_state"],
        # Source: ACDD
        # Specifies type of creator with one of the following: 'person', 'group',
        # 'institution', or 'position'. If this attribute is not specified, the
        # creator is assumed to be a person.
        "creator_type": config_global_attrs["creator_type"],
        # Source: ACDD, IOOS
        # The URL of the person (or other creator type specified by the creator_type
        # attribute) principally responsible for creating this data.
        "creator_url": config_global_attrs["creator_url"],
        # Source: ACDD
        # The name of any individuals, projects, or institutions that contributed to
        # the creation of this data.
        "contributor_name": config_global_attrs["contributor_name"],
        # Source: ACDD
        # The role of any individuals, projects, or institutions that contributed to
        # the creation of this data.
        "contributor_role": config_global_attrs["contributor_role"],
        # Source: IOOS
        # The URL of the controlled vocabulary used for the contributor_role attribute.
        # The default is "https://vocab.nerc.ac.uk/collection/G04/current/".
        "contributor_role_vocabulary": config_global_attrs[
            "contributor_role_vocabulary"
        ],
        # Source: IOOS
        # The URL of the individuals or institutions that contributed to the creation
        # of this data.
        "contributor_url": config_global_attrs["contributor_url"],
        # Source: Global
        # A string used to indicate the level of processing of the output data. It should be
        # formatted as a letter followed by a number. Typical values for this include:
        # a1 - data is ingested (no qc), b1 - data is ingested and quality checks applied,
        # c1 (or higher) - one or more a* or b* datastreams used to create a higher-level
        # data product. Only lowercase alphanumeric characters are allowed.
        "data_level": data_level,
        # Source: Global
        # A string used to identify the data being produced. Ideally resembles a shortened
        # lowercase version of the title. Only lowercase alphanumeric characters and '_'
        # are allowed.
        "dataset_name": location["output_name"].lower(),
        # Source: Global
        # Typically used as a label that uniquely identifies this data product from any
        # other data product. For file-based storage systems, the datastream attribute
        # is typically used to generate directory structures.
        # :datastream = cpr.wave_stats.c1 ;
        "datastream": f"{location['output_name']}.{location['output_name']}.{data_level}",
        # Source: Global, ACDD
        # A user-friendly description of the dataset. It should provide enough context
        # about the data for new users to quickly understand how the data can be used.
        "description": location["description"],
        # Source: Global
        # The DOI that has been registered for this dataset, if applicable.
        "doi": config_global_attrs["doi"],
        # Source: Global, ACDD, IOOS
        # CF attribute for identifying the featureType.
        "featureType": config_global_attrs["featureType"],
        # Source: ACDD
        # Describes the data's 2D or 3D geospatial extent in OGC's Well-Known Text (WKT)
        # Geometry format.
        "geospatial_bounds": geospatial_bounds["geospatial_bounds"],
        # Source: ACDD
        # The coordinate reference system (CRS) of the point coordinates in the
        # geospatial_bounds attribute.
        "geospatial_bounds_crs": geospatial_bounds["geospatial_bounds_crs"],
        # Source: ACDD
        # Describes a simple upper latitude limit; Geospatial_lat_max specifies the
        # northernmost latitude covered by the dataset.
        "geospatial_lat_max": geospatial_bounds["geospatial_lat_max"],
        # Source: ACDD
        # Describes a simple lower latitude limit; Geospatial_lat_min specifies the
        # southernmost latitude covered by the dataset.
        "geospatial_lat_min": geospatial_bounds["geospatial_lat_min"],
        # Source: ACDD
        # Units for the latitude axis described in 'geospatial_lat_min' and
        # 'geospatial_lat_max' attributes.
        "geospatial_lat_units": geospatial_bounds["geospatial_lat_units"],
        # Source: ACDD
        # Information about the targeted spacing of points in latitude.
        "geospatial_lat_resolution": geospatial_bounds["geospatial_lat_resolution"],
        # Source: ACDD
        # Describes a simple longitude limit; geospatial_lon_max specifies the
        # easternmost longitude covered by the dataset.
        "geospatial_lon_max": geospatial_bounds["geospatial_lon_max"],
        # Source: ACDD
        # Describes a simple longitude limit; geospatial_lon_min specifies the
        # westernmost longitude covered by the dataset.
        "geospatial_lon_min": geospatial_bounds["geospatial_lon_min"],
        # Source: ACDD
        # Units for the longitude axis described in 'geospatial_lon_min' and
        # 'geospatial_lon_max' attributes.
        "geospatial_lon_units": geospatial_bounds["geospatial_lon_units"],
        # Source: ACDD
        # Information about the targeted spacing of points in longitude.
        "geospatial_lon_resolution": geospatial_bounds["geospatial_lon_resolution"],
        # Source: ACDD
        # The vertical coordinate reference system (CRS) for the Z axis of the point
        # coordinates in the geospatial_bounds attribute.
        "geospatial_bounds_vertical_crs": vertical_attributes[
            "geospatial_bounds_vertical_crs"
        ],
        # Source: ACDD
        # Describes the numerically larger vertical limit.
        "geospatial_vertical_max": vertical_attributes["geospatial_vertical_max"],
        # Source: ACDD
        # Describes the numerically smaller vertical limit.
        "geospatial_vertical_min": vertical_attributes["geospatial_vertical_min"],
        # Source: ACDD
        # Describes the vertical origin
        "geospatial_vertical_origin": vertical_attributes["geospatial_vertical_origin"],
        # Source: ACDD
        # One of 'up' or 'down'. If up, vertical values are interpreted as 'altitude'.
        # If down, vertical values are interpreted as 'depth'.
        "geospatial_vertical_positive": vertical_attributes[
            "geospatial_vertical_positive"
        ],
        # Source: ACDD
        # Information about the targeted vertical spacing of points.
        "geospatial_vertical_resolution": vertical_attributes[
            "geospatial_vertical_resolution"
        ],
        # Source: ACDD
        # Units for the vertical axis.
        # "geospatial_vertical_units": "meters",
        "geospatial_vertical_units": vertical_attributes["geospatial_vertical_units"],
        # Source: Global
        # Attribute that will be recorded automatically by the pipeline.
        "history": get_history_string(),  # "Ran by jmcvey3 at 2024-03-28T15:41:04.600578",
        # Source: ACDD, IOOS
        # An identifier for the data set, provided by and unique within its naming
        # authority. The combination of the 'naming_authority' and the 'id' should
        # be globally unique.
        "id": f"{location['output_name']}.{config['dataset']['name']}.v{config['dataset']['version']}",
        # Source: IOOS
        # URL for background information about this dataset.
        "infoURL": config_global_attrs["infoURL"],
        # Source: Global
        "inputs": str(input_files),
        "input_history_json": input_history_json,
        # Source: ACDD
        # Name of the contributing instrument(s) or sensor(s) used to create this
        # data set or product.
        "instrument": None,
        # Source: ACDD
        # Controlled vocabulary for the names used in the 'instrument' attribute.
        "instrument_vocabulary": None,
        # Source: ACDD
        # A comma-separated list of key words and/or phrases.
        "keywords": config_global_attrs["keywords"],
        # Source: ACDD
        # If you are using a controlled vocabulary for the words/phrases in your
        # 'keywords' attribute, this is the unique name or identifier of the
        # vocabulary from which keywords are taken.
        "keywords_vocabulary": None,
        # Source: ACDD, IOOS
        # Provide the URL to a standard or specific license, enter 'Freely
        # Distributed' or 'None', or describe any restrictions to data access
        # and distribution in free text.
        "license": config_global_attrs["license"],
        # Source: Global
        # A label or acronym for the location where the data were obtained from.
        # Only alphanumeric characters and '_' are allowed.
        # "location_id": "cpr",
        # "make_model": "Sofar Spotter2",
        # Source: ACDD, IOOS
        # The organization that provides the initial id for the dataset.
        # "naming_authority": "gov.pnnl.sequim",
        "naming_authority": config_global_attrs["naming_authority"],
        # Source: ACDD, IOOS
        # Name of the platform(s) that supported the sensor data used to create this
        # data set or product.
        # "platform": "wave_buoy",
        # Source: IOOS
        # An optional, short identifier for the platform, if the data provider
        # prefers to define an id that differs from the dataset identifier.
        # "platform_id": None,
        # Source: IOOS
        # A descriptive, long name for the platform used in collecting the data.
        # "platform_name": None,
        # Source: ACDD, IOOS
        # Controlled vocabulary for the names used in the 'platform' attribute.
        # "platform_vocabulary": "http://mmisw.org/ont/ioos/platform",
        # Source: Global
        # Optional attribute used to cite other data, algorithms, etc. as needed.
        "references": config_global_attrs["references"],
        # Source: Global
        # An optional string which distinguishes these data from other datasets
        # produced by the same instrument.
        # "qualifier": None,
        # Source: Global
        # An optional string which describes the temporal resolution of the data.
        "temporal": location["temporal_resolution"],
        # Source: ACDD
        # The data type, as derived from Unidata's Common Data Model Scientific Data types.
        # "cdm_data_type": None,
        # Source: ACDD
        # Miscellaneous information about the data, not captured elsewhere.
        "comment": config_global_attrs["comment"],
        # Source: ACDD
        # The date on which this version of the data was created.
        "date_created": modification_dates["date_created"],
        # Source: ACDD
        # The date on which this data was formally issued.
        "date_issued": config["dataset"]["issue_date"],
        # Source: ACDD
        # The date on which the metadata was last modified.
        "date_metadata_modified": modification_dates["date_metadata_modified"],
        # Source: ACDD
        # The date on which the data was last modified.
        "date_modified": modification_dates["date_modified"],
        # Source: ACDD
        # A URL that gives the location of more complete metadata.
        "metadata_link": config_global_attrs["metadata_link"],
        # Source: ACDD
        # A textual description of the processing (or quality control) level
        # of the data.
        "processing_level": data_level,
        # Source: ACDD
        # Version identifier of the data file or product as assigned by the
        # data creator.
        "product_version": config["dataset"]["version"],
        # Source: ACDD
        # The overarching program(s) of which the dataset is a part.
        "program": config_global_attrs["program"],
        # Source: ACDD
        # The name of the project(s) principally responsible for originating
        # this data.
        "project": config_global_attrs["project"],
        # Source: ACDD
        # A paragraph describing the dataset, analogous to an abstract for a paper.
        "summary": location["summary"],
        # Source: IOOS
        # Country of the person or organization that distributes the data.
        "publisher_country": config_global_attrs["publisher_country"],
        # Source: ACDD, IOOS
        # The email address of the person responsible for publishing the data file
        # or product to users.
        "publisher_email": config_global_attrs["publisher_email"],
        # Source: ACDD, IOOS
        # The institution of the publisher.
        "publisher_institution": config_global_attrs["publisher_institution"],
        # Source: ACDD
        # The name of the person responsible for publishing the data file or
        # product to users.
        "publisher_name": config_global_attrs["publisher_name"],
        # Source: IOOS
        # State or province of the person or organization that distributes the data.
        "publisher_state": config_global_attrs["publisher_state"],
        # Source: ACDD
        # Specifies type of publisher with one of the following: 'person', 'group',
        # 'institution', or 'position'.
        "publisher_type": config_global_attrs["publisher_type"],
        # Source: ACDD, IOOS
        # The URL of the person responsible for publishing the data file or product
        # to users.
        "publisher_url": config_global_attrs["publisher_url"],
        # Source: ACDD
        # The method of production of the original data. If it was model-generated,
        # source should name the model and its version.
        "source": config["model_specification"]["model_version"],
    }

    # Remove key/value pairs where the value is None
    standardized_attributes = {
        k: v for k, v in standardized_attributes.items() if v is not None
    }

    if input_ds_is_original_model_output is True:
        original_attributes = {
            f"original_{key}": value for key, value in ds.attrs.items()
        }

        standardized_attributes.update(original_attributes)

    else:
        allowed_changing_keys = [
            "data_level",
            "datastream",
            "history",
            "id",
            "inputs",
            "temporal",
            "date_modified",
            "date_metadata_modified",
            "processing_level",
        ]

        # Raise an error if there are unexpected attribute changes
        validate_attribute_changes(ds, standardized_attributes, allowed_changing_keys)

    ds.attrs = standardized_attributes

    return ds


if __name__ == "__main__":
    # print(standardize_dataset_global_attrs(None, None, None, "a1"))
    import xarray as xr
    from pathlib import Path

    import sys

    sys.path.append("..")

    from config import config
    import coord_manager

    nc_file = Path("../data/00_raw/MD_AIS_west_hrBathy_0240.nc")
    print(f"Reading {nc_file} into ds...")
    ds = xr.open_dataset(nc_file, decode_times=False)
    print(ds.attrs)
    # std_coords = coord_manager.standardize_fvcom_coords(ds, None)
    # ds["latc"].values = std_coords["lat_centers"]
    # ds["lonc"].values = std_coords["lon_centers"]
    # ds["lat"].values = std_coords["lat_corners"]
    # ds["lon"].values = std_coords["lon_corners"]

    ds = ds.rename(
        {
            "lat": "latitude",
            "lon": "longitude",
            "latc": "latitude_center",
            "lonc": "longitude_center",
        }
    )

    print(f"Standardizing attrs from {nc_file.name} into ds...")
    standardize_dataset_global_attrs(
        ds,
        config,
        config["location_specification"]["cook_inlet"],
        "a1",
        [str(nc_file)],
        input_ds_is_original_model_output=True,
        coordinate_reference_system_string="CRS_String",
    )
    print("Done!")
