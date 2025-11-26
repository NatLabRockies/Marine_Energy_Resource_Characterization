"""
Generate manifest files for parquet partition datasets to enable efficient spatial queries.

This script scans parquet partition directories and creates a self-documenting JSON manifest
with the following features:

1. **Versioning**: Manifest version auto-increments on regeneration; data versions tracked per location
2. **Self-documenting schema**: Embedded schema section describes all fields for non-Python clients
3. **Spatial indexing**: Grid centroids enable KDTree-based spatial queries
4. **Path reconstruction**: Template-based file path generation from point data
5. **S3 integration**: Storage URIs for cloud-based data access

Output Structure:
    manifests/v{version}/
    ├── manifest_{version}.json     # Main index with schema, grid centroids, location metadata
    └── grids/                      # Grid detail files organized by lat/lon
        └── lat_{deg}/
            └── lon_{deg}/
                └── {grid_id}.json  # Per-grid point arrays

Usage:
    python generate_parquet_partition_manifest_json.py

The manifest is designed to be self-contained so that client libraries in any language can:
1. Download the manifest
2. Parse the schema to understand the structure
3. Query a point and reconstruct the path to the parquet file
"""

import json
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
import xarray as xr

from config import config
from src import file_manager


def parse_parquet_filename(filename):
    """
    Parse parquet filename to extract face ID, lat, lon, and temporal resolution.

    This parser mirrors the logic in file_name_convention_manager.generate_filename_for_data_level()
    and vap_simple_create_parquet_all_time_partition.get_partition_file_name().

    Expected format (from generate_filename_for_data_level):
    {location}.{dataset_name-temporal}.{data_level}.{date}.{time}.{ext}

    Where dataset_name is built as:
    {dataset}.face={faceid}.lat={lat}.lon={lon}
    and temporal is appended with '-' (hyphen), not '.'

    Full format:
    {location}.{dataset}.face={faceid}.lat={lat}.lon={lon}-{temporal}.b1.{date}.{time}.v{version}.parquet

    Examples:
    AK_cook_inlet.wpto_high_res_tidal.face=000123.lat=59.1234567.lon=-152.7890123-1h.b1.20050101.000000.v1.0.0.parquet
    AK_aleutian_islands.wpto_high_res_tidal.face=000299.lat=49.9379654.lon=-174.9613647-1h.b1.20100603.000000.v1.0.0.parquet

    Parameters
    ----------
    filename : str
        The parquet filename

    Returns
    -------
    dict
        Dictionary with keys: face_id, lat, lon, temporal, location, dataset

    Raises
    ------
    ValueError
        If filename doesn't match expected pattern
    """
    # Step 1: Extract temporal string first (must be either "1h" or "30m")
    # Search for -1h or -30m in the filename
    temporal = None
    if "-1h." in filename:
        temporal = "1h"
    elif "-30m." in filename:
        temporal = "30m"
    else:
        raise ValueError(
            f"Could not find required temporal string (-1h or -30m) in filename: {filename}"
        )

    # Step 2: Split by '.' to get components
    parts = filename.split(".")

    if len(parts) < 9:
        raise ValueError(
            f"Filename has too few components (expected >= 9, got {len(parts)}): {filename}"
        )

    # Step 3: Extract location and dataset
    location = parts[0]
    dataset = parts[1]

    # Step 4: Find face=, lat=, and lon= component indices
    face_part = None
    lat_start_idx = None
    lon_start_idx = None

    for i, part in enumerate(parts):
        if part.startswith("face="):
            face_part = part
        elif part.startswith("lat="):
            lat_start_idx = i
        elif part.startswith("lon="):
            lon_start_idx = i

    if not face_part or lat_start_idx is None or lon_start_idx is None:
        raise ValueError(
            f"Missing required components (face=, lat=, lon=) in filename: {filename}"
        )

    # Step 5: Parse face ID
    face_id = int(face_part.replace("face=", ""))

    # Step 6: Parse latitude - reconstruct from parts until we hit lon=
    # The lat spans from lat_start_idx until lon_start_idx
    lat_parts = []
    for i in range(lat_start_idx, lon_start_idx):
        lat_parts.append(parts[i])

    # Join with '.' and remove 'lat=' prefix
    lat_str = ".".join(lat_parts).replace("lat=", "")
    lat = float(lat_str)

    # Step 7: Parse longitude - reconstruct from parts, stop at temporal separator
    # The lon spans from lon_start_idx until we hit -{temporal}
    lon_parts = []
    for i in range(lon_start_idx, len(parts)):
        part = parts[i]

        # Check if this part ends with the temporal separator
        if part.endswith(f"-{temporal}"):
            # Remove the temporal suffix and add this final part
            lon_parts.append(part[: -len(f"-{temporal}")])
            break
        else:
            lon_parts.append(part)

    # Join with '.' and remove 'lon=' prefix
    lon_str = ".".join(lon_parts).replace("lon=", "")
    lon = float(lon_str)

    return {
        "location": location,
        "dataset": dataset,
        "face_id": face_id,
        "lat": lat,
        "lon": lon,
        "temporal": temporal,
    }


def parse_partition_path(partition_path):
    """
    Parse partition directory path to extract lat/lon degree and decimal components.

    Expected format:
    lat_deg={DD}/lon_deg={DD}/lat_dec={DD}/lon_dec={DD}

    Example:
    lat_deg=59/lon_deg=-153/lat_dec=12/lon_dec=78

    Parameters
    ----------
    partition_path : str or Path
        The partition directory path

    Returns
    -------
    dict or None
        Dictionary with keys: lat_deg, lon_deg, lat_dec, lon_dec
        Returns None if path doesn't match expected pattern
    """
    path_str = str(partition_path)
    parts = path_str.split("/")

    # Look for the partition components in the path
    result = {}
    for part in parts:
        if part.startswith("lat_deg="):
            result["lat_deg"] = int(part.split("=")[1])
        elif part.startswith("lon_deg="):
            result["lon_deg"] = int(part.split("=")[1])
        elif part.startswith("lat_dec="):
            result["lat_dec"] = int(part.split("=")[1])
        elif part.startswith("lon_dec="):
            result["lon_dec"] = int(part.split("=")[1])

    # Return None if we didn't find all components
    if len(result) == 4:
        return result
    return None


def get_parquet_file_list(partition_dir, location_name, config, use_cache=True):
    """
    Get list of parquet files, using cache if available.

    Parameters
    ----------
    partition_dir : Path
        Root directory containing parquet partitions
    location_name : str
        Name of location for cache file naming
    config : dict
        Configuration dictionary
    use_cache : bool
        Whether to use cached file list (default: True)

    Returns
    -------
    list
        List of Path objects for parquet files
    """
    partition_dir = Path(partition_dir)

    # Create cache directory in current working directory
    cache_dir = Path.cwd() / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Cache filename with version
    version = config["dataset"]["version"]
    cache_file = cache_dir / f"{location_name}_file_list.v{version}.parquet"

    # Try to load from cache
    if use_cache and cache_file.exists():
        print(f"Loading file list from cache: {cache_file}")
        df_cache = pd.read_parquet(cache_file)
        parquet_files = [Path(p) for p in df_cache["file_path"].tolist()]
        print(f"Loaded {len(parquet_files)} files from cache")
        return sorted(parquet_files)

    # Cache miss or disabled - scan filesystem
    print(f"Scanning directory for parquet files: {partition_dir}")
    parquet_files = sorted(partition_dir.rglob("*.parquet"))
    print(f"Found {len(parquet_files)} parquet files")

    # Save to cache
    print(f"Saving file list to cache: {cache_file}")
    df_cache = pd.DataFrame({"file_path": [str(p) for p in parquet_files]})
    df_cache.to_parquet(cache_file, index=False)
    print("Cache saved")

    return parquet_files


def scan_parquet_partitions(partition_dir, location_name, config, use_cache=True):
    """
    Scan parquet partition directory and collect metadata for all files.
    Uses pandas apply for fast batch processing.

    Parameters
    ----------
    partition_dir : Path
        Root directory containing parquet partitions
    location_name : str
        Name of location for cache file naming
    config : dict
        Configuration dictionary
    use_cache : bool
        Whether to use cached file list (default: True)

    Returns
    -------
    list
        List of dictionaries containing file metadata
    """
    partition_dir = Path(partition_dir)

    if not partition_dir.exists():
        raise ValueError(f"Partition directory does not exist: {partition_dir}")

    print(f"Scanning directory: {partition_dir}")

    # Get parquet files (from cache or filesystem scan)
    parquet_files = get_parquet_file_list(
        partition_dir, location_name, config, use_cache=use_cache
    )

    # Build dataframe with file paths
    print("Building file path dataframe...")
    df = pd.DataFrame({"file_path": parquet_files})

    # Extract basic file info
    print("Extracting file metadata...")
    df["filename"] = df["file_path"].apply(lambda x: x.name)
    df["relative_path"] = df["file_path"].apply(
        lambda x: str(x.relative_to(partition_dir))
    )
    df["parent_path"] = df["file_path"].apply(
        lambda x: str(x.relative_to(partition_dir).parent)
    )

    print(f"Created dataframe with {len(df)} files")

    # Parse all filenames using apply
    print("Parsing filenames...")

    def safe_parse_filename(filename):
        try:
            return pd.Series(parse_parquet_filename(filename))
        except ValueError:
            print(f"\nERROR: Could not parse filename: {filename}")
            raise

    parsed_df = df["filename"].apply(safe_parse_filename)
    df = pd.concat([df, parsed_df], axis=1)
    print("  Filename parsing complete")

    # Parse partition paths using apply
    print("Parsing partition paths...")

    def safe_parse_partition(parent_path):
        partition_info = parse_partition_path(parent_path)
        if partition_info is None:
            print(f"\nERROR: Could not parse partition path: {parent_path}")
            raise ValueError(f"Failed to parse partition path: {parent_path}")
        return pd.Series(partition_info)

    partition_df = df["parent_path"].apply(safe_parse_partition)
    partition_df.columns = ["lat_deg", "lon_deg", "lat_dec", "lon_dec"]
    print("  Partition path parsing complete")

    # Build final metadata list
    print("Building final metadata list...")
    file_metadata = []
    for idx, row in df.iterrows():
        if idx % 50000 == 0 and idx > 0:
            print(f"  Assembled {idx}/{len(df)} records...")

        metadata = {
            "face_id": int(row["face_id"]),
            "lat": float(row["lat"]),
            "lon": float(row["lon"]),
            "location": row["location"],
            "dataset": row["dataset"],
            "temporal": row["temporal"],
            "partition": {
                "lat_deg": int(partition_df.loc[idx, "lat_deg"]),
                "lon_deg": int(partition_df.loc[idx, "lon_deg"]),
                "lat_dec": int(partition_df.loc[idx, "lat_dec"]),
                "lon_dec": int(partition_df.loc[idx, "lon_dec"]),
            },
            "file_path": row["relative_path"],
        }
        file_metadata.append(metadata)

    print(f"Successfully processed {len(file_metadata)} files")
    return file_metadata


def build_compact_grid_index(file_metadata, config):
    """
    Build compact grid index with centroids and point counts.

    Creates ultra-compact grid detail files with just string arrays for points.
    All path reconstruction metadata is stored in the main manifest.

    Parameters
    ----------
    file_metadata : list
        List of file metadata dictionaries
    config : dict
        Configuration dictionary

    Returns
    -------
    tuple
        (grid_index_list, grid_details_dict)
        - grid_index_list: List of grid metadata for main manifest
        - grid_details_dict: Dict mapping grid_id to compact point data
    """
    decimal_places = config["partition"]["decimal_places"]
    grid_resolution = 1.0 / (10**decimal_places)  # 0.01 for decimal_places=2

    # Group points by grid
    grid_groups = {}

    for metadata in file_metadata:
        partition = metadata["partition"]

        # Create grid ID
        grid_id = f"{partition['lat_deg']}_{partition['lon_deg']}_{partition['lat_dec']}_{partition['lon_dec']}"

        if grid_id not in grid_groups:
            grid_groups[grid_id] = {
                "lat_deg": partition["lat_deg"],
                "lon_deg": partition["lon_deg"],
                "lat_dec": partition["lat_dec"],
                "lon_dec": partition["lon_dec"],
                "points": [],
            }

        # Store as string array: [lat_str, lon_str, face_id_str]
        # face_id from metadata is already an int, format as zero-padded string
        # Use index_max_digits from config (default 8) for padding
        index_max_digits = config["partition"]["index_max_digits"]
        face_id_str = f"{metadata['face_id']:0{index_max_digits}d}"
        lat_str = f"{metadata['lat']:.7f}"  # Match coord_digits_max from config
        lon_str = f"{metadata['lon']:.7f}"

        grid_groups[grid_id]["points"].append([lat_str, lon_str, face_id_str])

        # Track location for this grid (all points in a grid should have same location)
        if "location" not in grid_groups[grid_id]:
            grid_groups[grid_id]["location"] = metadata["location"]

    # Build grid index and details
    grid_index = []
    grid_details = {}

    for grid_id, group in grid_groups.items():
        # Calculate grid bounds
        # IMPORTANT: Partition encoding uses int(coord) for deg and int(abs(coord*100) % 100) for dec
        # Example: lon=-70.70 gives lon_deg=-70, lon_dec=70
        # To reconstruct: deg is already signed, dec is always positive offset
        # For negative: -70 - 0.70 = -70.70 ✓
        # For positive: 43 + 0.05 = 43.05 ✓
        lat_deg = group["lat_deg"]
        lat_dec = group["lat_dec"]
        lon_deg = group["lon_deg"]
        lon_dec = group["lon_dec"]

        # Reconstruct coordinate from degree and decimal parts
        # Sign of deg determines whether to add or subtract the decimal part
        lat_min = (
            lat_deg + lat_dec / (10**decimal_places)
            if lat_deg >= 0
            else lat_deg - lat_dec / (10**decimal_places)
        )
        lon_min = (
            lon_deg + lon_dec / (10**decimal_places)
            if lon_deg >= 0
            else lon_deg - lon_dec / (10**decimal_places)
        )
        lat_max = lat_min + grid_resolution
        lon_max = lon_min + grid_resolution

        # Calculate centroid
        centroid_lat = (lat_min + lat_max) / 2
        centroid_lon = (lon_min + lon_max) / 2

        # Grid metadata for main manifest (centroids only)
        grid_index.append(
            {
                "id": grid_id,
                "lat_deg": group["lat_deg"],
                "lon_deg": group["lon_deg"],
                "lat_dec": group["lat_dec"],
                "lon_dec": group["lon_dec"],
                "bounds": [lat_min, lat_max, lon_min, lon_max],
                "centroid": [centroid_lat, centroid_lon],
                "n": len(group["points"]),
            }
        )

        # Compact grid detail file - just string arrays plus location for path reconstruction
        grid_details[grid_id] = {
            "grid_id": grid_id,
            "location": group["location"],
            "points_columns": ["lat", "lon", "face_id"],
            "points": group["points"],
        }

    return grid_index, grid_details


def build_faceid_index(file_metadata):
    """
    Build face ID index for direct face lookup.

    Parameters
    ----------
    file_metadata : list
        List of file metadata dictionaries

    Returns
    -------
    dict
        Face ID index mapping location/face_id to file info
    """
    faceid_index = {}

    for metadata in file_metadata:
        location = metadata["location"]
        face_id = metadata["face_id"]

        # Create hierarchical key: location -> face_id
        if location not in faceid_index:
            faceid_index[location] = {}

        faceid_index[location][face_id] = {
            "lat": metadata["lat"],
            "lon": metadata["lon"],
            "file_path": metadata["file_path"],
        }

    return faceid_index


def extract_geospatial_bounds_from_nc(config, location):
    """
    Extract geospatial bounds from a b1_vap NetCDF file.

    Loads one representative NC file and extracts the geospatial_* attributes
    that were computed by compute_geospatial_bounds() in attrs_manager.py.

    Parameters
    ----------
    config : dict
        Configuration dictionary
    location : dict
        Location specification dictionary

    Returns
    -------
    dict or None
        Dictionary with geospatial bounds attributes, or None if file not found
    """
    # Get b1_vap directory
    vap_dir = file_manager.get_vap_output_dir(config, location)

    if not vap_dir.exists():
        print(f"  Warning: VAP directory does not exist: {vap_dir}")
        return None

    # Find first NC file
    nc_files = sorted(vap_dir.glob("*.nc"))
    if not nc_files:
        print(f"  Warning: No NC files found in: {vap_dir}")
        return None

    nc_file = nc_files[0]
    print(f"  Loading geospatial bounds from: {nc_file.name}")

    try:
        # Open dataset and extract geospatial attributes
        with xr.open_dataset(nc_file, decode_times=False) as ds:
            # Extract all geospatial_* attributes
            geospatial_attrs = {}
            for key, value in ds.attrs.items():
                if key.startswith("geospatial_"):
                    geospatial_attrs[key] = value

            if not geospatial_attrs:
                print(f"  Warning: No geospatial_* attributes found in {nc_file.name}")
                return None

            return geospatial_attrs

    except Exception as e:
        print(f"  Error loading {nc_file.name}: {e}")
        return None


def increment_manifest_version(current_version):
    """
    Increment the patch version of a semantic version string.

    Parameters
    ----------
    current_version : str
        Current version string (e.g., "1.0.0")

    Returns
    -------
    str
        Incremented version string (e.g., "1.0.1")
    """
    parts = current_version.split(".")
    if len(parts) != 3:
        raise ValueError(f"Invalid semver format: {current_version}")
    major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])
    return f"{major}.{minor}.{patch + 1}"


def load_existing_manifest(output_dir):
    """
    Load existing manifest if it exists to preserve version history.

    Parameters
    ----------
    output_dir : Path
        Directory containing existing manifest

    Returns
    -------
    dict or None
        Existing manifest data, or None if not found
    """
    manifest_files = sorted(output_dir.glob("manifest_*.json"))
    if not manifest_files:
        return None

    # Load the most recent manifest (sorted alphabetically, so latest version last)
    latest_manifest = manifest_files[-1]
    print(f"  Found existing manifest: {latest_manifest.name}")

    with open(latest_manifest, "r") as f:
        return json.load(f)


def build_manifest_schema():
    """
    Build the self-documenting schema section for the manifest.

    Returns
    -------
    dict
        Schema documentation dictionary
    """
    return {
        "spec_version": "Semantic version of the manifest format specification. Breaking changes increment major version.",
        "manifest_version": "Auto-incrementing version of this manifest file. Patch increments on each regeneration.",
        "manifest_generated": "ISO 8601 timestamp (UTC) when this manifest was generated.",
        "dataset": {
            "_description": "Dataset identification metadata",
            "name": "Internal dataset identifier used in file paths and programmatic access",
            "label": "Human-readable dataset name for display purposes",
        },
        "storage": {
            "_description": "Storage configuration for data access (S3 and HPC)",
            "s3_bucket": "AWS S3 bucket name where data is stored",
            "s3_prefix": "S3 key prefix path within the bucket",
            "s3_base_uri": "Full S3 URI base path for data files (s3://{bucket}/{prefix})",
            "s3_manifest_base_uri": "S3 URI directory where manifest versions are stored",
            "hpc_base_path": "Absolute filesystem path for HPC local access (NREL Kestrel)",
        },
        "partition": {
            "_description": "Spatial partitioning configuration for parquet files",
            "decimal_places": "Number of decimal places used in partition path encoding. For decimal_places=2: lat_dec = int(abs(lat * 100) % 100)",
            "coord_digits_max": "Maximum decimal precision for latitude/longitude values in filenames",
            "index_max_digits": "Zero-padding width for face_id in filenames (e.g., 8 means face=00001234)",
            "data_level": "Data processing level identifier used in file paths (e.g., 'b1_vap_by_point_partition')",
            "grid_resolution_deg": "Grid cell size in degrees, derived as 1.0 / (10 ** decimal_places)",
        },
        "path_template": {
            "_description": "Template string for reconstructing parquet file paths from point data",
            "template": "The path template string with {placeholder} variables",
            "placeholders": {
                "location": "Location output_name (e.g., 'AK_cook_inlet')",
                "data_version": "Data version for the location (e.g., '1.0.0'). Path uses 'v' prefix (v1.0.0)",
                "data_level": "Data processing level (e.g., 'b1_vap_by_point_partition')",
                "lat_deg": "Integer latitude degrees (signed, e.g., 59 or -70)",
                "lon_deg": "Integer longitude degrees (signed, e.g., -152)",
                "lat_dec": "Latitude decimal component, zero-padded (e.g., '05' for 0.05°)",
                "lon_dec": "Longitude decimal component, zero-padded (e.g., '78' for 0.78°)",
                "face_id": "Zero-padded face identifier string",
                "lat": "Full precision latitude string from point data",
                "lon": "Full precision longitude string from point data",
                "temporal": "Temporal resolution code ('1h' for hourly, '30m' for half-hourly)",
                "date": "Start date in YYYYMMDD format",
                "time": "Start time in HHMMSS format",
            },
        },
        "locations": {
            "_description": "Per-location metadata keyed by output_name",
            "_key_format": "Location output_name (e.g., 'AK_cook_inlet')",
            "label": "Human-readable location name for display",
            "latest_version": "String indicating the latest/recommended data version for this location",
            "versions": {
                "_description": "Object mapping data version strings to version metadata",
                "_key_format": "Semantic version string (e.g., '1.0.0')",
                "release_date": "ISO 8601 date when this version was released",
            },
            "point_count": "Total number of data points (faces) for this location",
            "temporal": "Temporal resolution code ('1h' or '30m')",
            "date": "Dataset start date in YYYYMMDD format",
            "time": "Dataset start time in HHMMSS format",
            "geospatial_lat_min": "Minimum latitude of location bounding box (degrees_north)",
            "geospatial_lat_max": "Maximum latitude of location bounding box (degrees_north)",
            "geospatial_lon_min": "Minimum longitude of location bounding box (degrees_east)",
            "geospatial_lon_max": "Maximum longitude of location bounding box (degrees_east)",
            "geospatial_bounds": "WKT POLYGON string defining the location boundary",
        },
        "grid_centroids": {
            "_description": "Arrays of grid cell centroid coordinates for spatial indexing",
            "lat": "Array of grid cell centroid latitudes (degrees_north)",
            "lon": "Array of grid cell centroid longitudes (degrees_east), same length as lat",
        },
        "grid_details_path": "Relative path from manifest to grid detail files directory",
        "total_grids": "Total number of grid cells across all locations",
        "total_points": "Total number of data points (faces) across all locations",
    }


def generate_compact_manifest(config, output_dir, existing_manifest=None):
    """
    Generate compact two-tier manifest structure with spec v2.0.0:
    - Main manifest_{version}.json with grid metadata, schema, and path template
    - Individual grid detail JSON files in grids/ subdirectory

    Parameters
    ----------
    config : dict
        Configuration dictionary
    output_dir : Path
        Output directory for manifest files
    existing_manifest : dict, optional
        Existing manifest to merge version history from

    Returns
    -------
    dict
        Generated manifest dictionary
    """
    print("\n=== Generating Compact Manifest (Spec v2.0.0) ===")

    # Determine manifest version
    if existing_manifest and "manifest_version" in existing_manifest:
        old_version = existing_manifest["manifest_version"]
        manifest_version = increment_manifest_version(old_version)
        print(f"  Incrementing manifest version: {old_version} -> {manifest_version}")
    else:
        manifest_version = config["manifest"]["version"]
        print(f"  Starting new manifest version: {manifest_version}")

    # Get dataset version from config
    dataset_version = config["dataset"]["version"]
    dataset_issue_date = config["dataset"]["issue_date"]

    # Collect file metadata from all locations
    all_file_metadata = []
    location_data = {}  # Per-location data structure

    for location_key, location in config["location_specification"].items():
        print(f"\nProcessing location: {location['label']} ({location_key})")

        partition_dir = file_manager.get_vap_partition_output_dir(config, location)

        if not partition_dir.exists():
            print(f"  Partition directory does not exist, skipping: {partition_dir}")
            continue

        file_metadata = scan_parquet_partitions(
            partition_dir, location["output_name"], config, use_cache=True
        )

        all_file_metadata.extend(file_metadata)

        # Extract geospatial bounds from b1_vap NC file
        print("  Extracting geospatial bounds from NC file...")
        geospatial_bounds = extract_geospatial_bounds_from_nc(config, location)

        # Parse start_date_utc to extract date and time components
        start_date_str = location["start_date_utc"]
        date_part, time_part = start_date_str.split(" ")
        date_formatted = date_part.replace("-", "")
        time_formatted = time_part.replace(":", "")

        # Determine temporal string from expected_delta_t_seconds
        expected_delta_t_seconds = location["expected_delta_t_seconds"]
        temporal_mapping = {3600: "1h", 1800: "30m"}
        if expected_delta_t_seconds not in temporal_mapping:
            raise ValueError(
                f"Unexpected expected_delta_t_seconds configuration {expected_delta_t_seconds}"
            )
        temporal_string = temporal_mapping[expected_delta_t_seconds]

        # Build version history for this location
        # Start with existing versions if available
        location_name = location["output_name"]
        existing_versions = {}
        if (
            existing_manifest
            and "locations" in existing_manifest
            and location_name in existing_manifest["locations"]
            and "versions" in existing_manifest["locations"][location_name]
        ):
            existing_versions = existing_manifest["locations"][location_name][
                "versions"
            ]

        # Add/update current version
        versions = dict(existing_versions)
        versions[dataset_version] = {
            "release_date": dataset_issue_date,
        }

        # Store location data with versioning
        location_data[location_name] = {
            "label": location["label"],
            "latest_version": dataset_version,
            "versions": versions,
            "point_count": len(file_metadata),
            "date": date_formatted,
            "time": time_formatted,
            "temporal": temporal_string,
        }

        # Add geospatial bounds if available
        if geospatial_bounds:
            location_data[location_name].update(geospatial_bounds)

    print(f"\nTotal files across all locations: {len(all_file_metadata)}")

    # Build compact grid index
    print("\nBuilding compact grid index...")
    grid_index, grid_details = build_compact_grid_index(all_file_metadata, config)
    print(f"  Created {len(grid_index)} grid cells")
    print(f"  Created {len(grid_details)} grid detail files")

    # Extract grid centroids
    spec_decimal_places = config["partition"]["decimal_places"]
    grid_lats = [round(g["centroid"][0], spec_decimal_places + 1) for g in grid_index]
    grid_lons = [round(g["centroid"][1], spec_decimal_places + 1) for g in grid_index]

    # Build path template with all configurable components
    # Note: version uses 'v' prefix (e.g., v1.0.0) to match directory structure
    # Note: face_id is passed as pre-padded string (e.g., "00126388")
    # Note: filename ends with .v{data_version}.parquet
    dec_format_spec = f"0{spec_decimal_places}d"
    dataset_name = config["dataset"]["name"]
    path_template = (
        "{location}/v{data_version}/{data_level}/"
        f"lat_deg={{lat_deg}}/lon_deg={{lon_deg}}/"
        f"lat_dec={{lat_dec:{dec_format_spec}}}/lon_dec={{lon_dec:{dec_format_spec}}}/"
        f"{{location}}.{dataset_name}.face={{face_id}}.lat={{lat}}.lon={{lon}}-{{temporal}}.b1.{{date}}.{{time}}.v{{data_version}}.parquet"
    )

    # Build storage configuration
    storage_config = config["storage"]
    s3_bucket = storage_config["s3_bucket"]
    s3_prefix = storage_config["s3_prefix"]

    # Create the manifest with new spec v2.0.0 structure
    manifest = {
        # Top-level versioning
        "spec_version": config["manifest"]["spec_version"],
        "manifest_version": manifest_version,
        "manifest_generated": datetime.utcnow().isoformat() + "Z",
        # Self-documenting schema
        "schema": build_manifest_schema(),
        # Dataset identification
        "dataset": {
            "name": config["dataset"]["name"],
            "label": config["dataset"]["label"],
        },
        # Storage configuration
        "storage": {
            "s3_bucket": s3_bucket,
            "s3_prefix": s3_prefix,
            "s3_base_uri": f"s3://{s3_bucket}/{s3_prefix}",
            "s3_manifest_base_uri": f"s3://{s3_bucket}/{s3_prefix}/manifest",
            "hpc_base_path": storage_config["hpc_base_path"],
        },
        # Partition configuration
        "partition": {
            "decimal_places": spec_decimal_places,
            "coord_digits_max": config["partition"]["coord_digits_max"],
            "index_max_digits": config["partition"]["index_max_digits"],
            "data_level": config["partition"]["data_level"],
            "grid_resolution_deg": 1.0 / (10**spec_decimal_places),
        },
        # Path template for file reconstruction
        "path_template": {
            "template": path_template,
            "example": "AK_cook_inlet/v1.0.0/b1_vap_by_point_partition/lat_deg=59/lon_deg=-152/lat_dec=12/lon_dec=78/AK_cook_inlet.wpto_high_res_tidal.face=00012345.lat=59.1234567.lon=-152.7890123-1h.b1.20050101.000000.v1.0.0.parquet",
        },
        # Summary statistics
        "total_grids": len(grid_index),
        "total_points": len(all_file_metadata),
        # Location data with versioning
        "locations": location_data,
        # Grid centroids for spatial indexing
        "grid_centroids": {
            "lat": grid_lats,
            "lon": grid_lons,
        },
        # Path to grid details (relative to manifest)
        "grid_details_path": "grids",
    }

    # Write main manifest with version in filename
    manifest_filename = f"manifest_{manifest_version}.json"
    manifest_file = output_dir / manifest_filename
    print(f"\nWriting main manifest to: {manifest_file}")
    with open(manifest_file, "w") as f:
        json.dump(manifest, f)

    file_size_mb = manifest_file.stat().st_size / (1024 * 1024)
    print(f"  Main manifest size: {file_size_mb:.2f} MB")

    # Create grids subdirectory
    grids_dir = output_dir / "grids"
    grids_dir.mkdir(parents=True, exist_ok=True)

    # Write individual grid detail files
    print(f"\nWriting {len(grid_details)} grid detail files to: {grids_dir}")
    print("  Organizing into lat_deg/lon_deg subdirectories...")

    for idx, (grid_id, details) in enumerate(grid_details.items()):
        if idx % 10000 == 0 and idx > 0:
            print(f"  Written {idx}/{len(grid_details)} grid files...")

        # Add dataset version to grid details
        details["data_version"] = dataset_version

        # Parse grid_id to extract lat_deg and lon_deg
        parts = grid_id.split("_")
        lat_deg = parts[0]
        lon_deg = parts[1]

        # Create nested directory structure
        lat_dir = grids_dir / f"lat_{lat_deg}"
        lon_dir = lat_dir / f"lon_{lon_deg}"
        lon_dir.mkdir(parents=True, exist_ok=True)

        # Write grid file
        grid_file = lon_dir / f"{grid_id}.json"
        with open(grid_file, "w") as f:
            json.dump(details, f, indent=2)

    print("  Completed writing all grid detail files")

    # Calculate total size
    total_size_mb = sum(f.stat().st_size for f in grids_dir.rglob("*.json")) / (
        1024 * 1024
    )
    total_size_mb += file_size_mb
    print(f"\nTotal manifest size: {total_size_mb:.2f} MB")
    print(f"  Main manifest: {file_size_mb:.2f} MB")
    print(f"  Grid details: {total_size_mb - file_size_mb:.2f} MB")

    return manifest


def main():
    """
    Main function to generate compact two-tier manifest with spec v2.0.0.

    Features:
    - Auto-increments manifest version if existing manifest found
    - Preserves version history for each location
    - Self-documenting schema embedded in manifest
    - Versioned grid detail files
    """
    print("=" * 80)
    print("Compact Parquet Partition Manifest Generation (Spec v2.0.0)")
    print("=" * 80)

    # Get output directory from file_manager (uses config["dir"]["output"]["manifest"])
    manifest_version = config["manifest"]["version"]
    output_dir = file_manager.get_manifest_output_dir(config, manifest_version)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nOutput directory: {output_dir}")

    # Check for existing manifest to preserve version history
    print("\nChecking for existing manifest...")
    existing_manifest = load_existing_manifest(output_dir)
    if existing_manifest:
        print("  Will merge version history from existing manifest")
    else:
        print("  No existing manifest found, starting fresh")

    # Generate compact manifest
    manifest = generate_compact_manifest(config, output_dir, existing_manifest)

    print("\n" + "=" * 80)
    print("Manifest generation complete!")
    print("=" * 80)
    print(f"\nSpec version: {manifest['spec_version']}")
    print(f"Manifest version: {manifest['manifest_version']}")
    print("\nGenerated structure:")
    print(f"  {output_dir}/")
    print(f"    ├── manifest_{manifest['manifest_version']}.json  (main index)")
    print(
        f"    └── grids/                        ({manifest['total_grids']} grid detail files)"
    )
    print("\nLocations with version info:")
    for loc_name, loc_data in manifest["locations"].items():
        versions = list(loc_data["versions"].keys())
        latest = loc_data["latest_version"]
        print(f"  {loc_name}: latest={latest}, all_versions={versions}")

    print("\nS3 URIs:")
    print(f"  Base: {manifest['storage']['s3_base_uri']}")
    print(
        f"  Manifest: {manifest['storage']['s3_manifest_base_uri']}/{manifest['manifest_version']}/manifest_{manifest['manifest_version']}.json"
    )

    print("\nUsage:")
    print("  - Load manifest_{version}.json for fast grid-level queries")
    print("  - Lazy-load individual grid files as needed")
    print("  - Use TidalManifestQuery class for spatial queries")
    print(
        "  - Use path_template with location's latest_version to construct file paths"
    )


if __name__ == "__main__":
    main()
