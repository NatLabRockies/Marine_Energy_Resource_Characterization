"""
Generate manifest files for parquet partition datasets to enable efficient spatial queries.

This script scans parquet partition directories and creates JSON manifests for:
1. Spatial lookups (by lat/lon coordinates)
2. Face ID lookups (by face identifier)

The manifests enable efficient point, line, and polygon queries without scanning the entire dataset.

Usage:
    python generate_parquet_partition_manifest_json.py

The script will generate:
    - high_res_tidal_point_manifest.json: Combined spatial and face ID manifest
    - high_res_tidal_spatial_manifest.json: Spatial grid index only
    - high_res_tidal_faceid_manifest.json: Face ID lookup only
"""

import json
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np

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
    {location}.{dataset}.face={faceid}.lat={lat}.lon={lon}-{temporal}.b4.{date}.{time}.parquet

    Examples:
    AK_cook_inlet.wpto_high_res_tidal.face=000123.lat=59.1234567.lon=-152.7890123-1h.b4.20050101.000000.parquet
    AK_aleutian_islands.wpto_high_res_tidal.face=000299.lat=49.9379654.lon=-174.9613647-1h.b4.20100603.000000.parquet

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

    # Step 4: Find face= and lat= components
    face_part = None
    lat_part = None
    lon_start_idx = None

    for i, part in enumerate(parts):
        if part.startswith("face="):
            face_part = part
        elif part.startswith("lat="):
            lat_part = part
        elif part.startswith("lon="):
            lon_start_idx = i

    if not face_part or not lat_part or lon_start_idx is None:
        raise ValueError(
            f"Missing required components (face=, lat=, lon=) in filename: {filename}"
        )

    # Step 5: Parse face ID
    face_id = int(face_part.replace("face=", ""))

    # Step 6: Parse latitude
    lat = float(lat_part.replace("lat=", ""))

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
        - grid_details_dict: Dict mapping grid_id to detailed point data
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
                "location": metadata["location"],
                "temporal": metadata["temporal"],
            }

        grid_groups[grid_id]["points"].append(
            {
                "face": metadata["face_id"],
                "lat": metadata["lat"],
                "lon": metadata["lon"],
                "file_path": metadata["file_path"],
            }
        )

    # Build grid index and details
    grid_index = []
    grid_details = {}

    for grid_id, group in grid_groups.items():
        # Calculate grid bounds
        lat_min = group["lat_deg"] + group["lat_dec"] / (10**decimal_places)
        lon_min = group["lon_deg"] + group["lon_dec"] / (10**decimal_places)
        lat_max = lat_min + grid_resolution
        lon_max = lon_min + grid_resolution

        # Calculate centroid
        centroid_lat = (lat_min + lat_max) / 2
        centroid_lon = (lon_min + lon_max) / 2

        # Grid metadata for main manifest
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
                "loc": group["location"],
                "temporal": group["temporal"],
            }
        )

        # Detailed point data for grid detail file
        grid_details[grid_id] = {
            "grid_id": grid_id,
            "location": group["location"],
            "temporal": group["temporal"],
            "points": [
                {
                    "face": p["face"],
                    "lat": p["lat"],
                    "lon": p["lon"],
                    "file_path": p["file_path"],
                }
                for p in group["points"]
            ],
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


def compute_bounds(file_metadata):
    """
    Compute spatial bounds of the dataset.

    Parameters
    ----------
    file_metadata : list
        List of file metadata dictionaries

    Returns
    -------
    dict
        Dictionary with lat_min, lat_max, lon_min, lon_max
    """
    if not file_metadata:
        return {"lat_min": None, "lat_max": None, "lon_min": None, "lon_max": None}

    lats = [m["lat"] for m in file_metadata]
    lons = [m["lon"] for m in file_metadata]

    return {
        "lat_min": min(lats),
        "lat_max": max(lats),
        "lon_min": min(lons),
        "lon_max": max(lons),
    }


def generate_compact_manifest(config, output_dir):
    """
    Generate compact two-tier manifest structure:
    - Main manifest.json with grid metadata
    - Individual grid detail JSON files in grids/ subdirectory

    Parameters
    ----------
    config : dict
        Configuration dictionary
    output_dir : Path
        Output directory for manifest files
    """
    print("\n=== Generating Compact Manifest ===")

    # Collect file metadata from all locations
    all_file_metadata = []
    location_stats = {}
    location_list = []

    for location_key, location in config["location_specification"].items():
        print(f"\nProcessing location: {location['label']} ({location_key})")

        partition_dir = file_manager.get_vap_partition_output_dir(config, location)

        if not partition_dir.exists():
            print(f"  Partition directory does not exist, skipping: {partition_dir}")
            continue

        file_metadata = scan_parquet_partitions(
            partition_dir, location["output_name"], config, use_cache=True
        )

        # Add location prefix to file paths
        for metadata in file_metadata:
            metadata["file_path"] = f"{location['output_name']}/{metadata['file_path']}"

        all_file_metadata.extend(file_metadata)

        location_stats[location_key] = {
            "label": location["label"],
            "output_name": location["output_name"],
            "file_count": len(file_metadata),
            "expected_face_count": location["face_count"],
        }

        # Track unique locations
        if location["output_name"] not in location_list:
            location_list.append(location["output_name"])

    print(f"\nTotal files across all locations: {len(all_file_metadata)}")

    # Build compact grid index
    print("\nBuilding compact grid index...")
    grid_index, grid_details = build_compact_grid_index(all_file_metadata, config)
    print(f"  Created {len(grid_index)} grid cells")
    print(f"  Created {len(grid_details)} grid detail files")

    # Compute bounds
    print("\nComputing spatial bounds...")
    bounds = compute_bounds(all_file_metadata)

    # Extract grid centroids for ultra-compact manifest
    # Round to 1 extra from the spec decimal places to avoid floating point precision bloat
    # (e.g., -65.95499999999998 -> -65.955)
    # Spec Grid resolution is 0.01°, centroids are at midpoint, so 3 decimals handles the centroid conversion
    spec_decimal_places = config["partition"]["decimal_places"]
    grid_lats = [round(g["centroid"][0], spec_decimal_places + 1) for g in grid_index]
    grid_lons = [round(g["centroid"][1], spec_decimal_places + 1) for g in grid_index]

    # Create ultra-compact main manifest (grid centroids only)
    manifest = {
        "version": config["dataset"]["version"],
        "decimal_places": spec_decimal_places,
        "grid_resolution_deg": 1.0 / (10 ** config["partition"]["decimal_places"]),
        "total_grids": len(grid_index),
        "total_points": len(all_file_metadata),
        "spatial_bounds": bounds,
        "metadata": {
            "dataset_name": config["dataset"]["name"],
            "dataset_label": config["dataset"]["label"],
            "manifest_generated": datetime.now().isoformat(),
            "location_stats": location_stats,
        },
        "grid_centroids": {
            "lat": grid_lats,
            "lon": grid_lons,
        },
    }

    # Write main manifest
    # Use compact JSON encoding (no indent) to keep file size small
    manifest_file = output_dir / "manifest.json"
    print(f"\nWriting main manifest to: {manifest_file}")
    with open(manifest_file, "w") as f:
        json.dump(manifest, f)

    file_size_mb = manifest_file.stat().st_size / (1024 * 1024)
    print(f"  Main manifest size: {file_size_mb:.2f} MB")

    # Create grids subdirectory
    grids_dir = output_dir / "grids"
    grids_dir.mkdir(parents=True, exist_ok=True)

    # Write individual grid detail files with nested directory structure
    # Organize by lat_deg/lon_deg to keep subdirectories manageable
    print(f"\nWriting {len(grid_details)} grid detail files to: {grids_dir}")
    print("  Organizing into lat_deg/lon_deg subdirectories...")

    for idx, (grid_id, details) in enumerate(grid_details.items()):
        if idx % 10000 == 0 and idx > 0:
            print(f"  Written {idx}/{len(grid_details)} grid files...")

        # Parse grid_id to extract lat_deg and lon_deg
        # Format: "lat_deg_lon_deg_lat_dec_lon_dec"
        parts = grid_id.split("_")
        lat_deg = parts[0]
        lon_deg = parts[1]

        # Create nested directory structure: grids/lat_deg/lon_deg/
        lat_dir = grids_dir / f"lat_{lat_deg}"
        lon_dir = lat_dir / f"lon_{lon_deg}"
        lon_dir.mkdir(parents=True, exist_ok=True)

        # Write grid file
        grid_file = lon_dir / f"{grid_id}.json"
        with open(grid_file, "w") as f:
            json.dump(details, f, indent=2)

    print("  Completed writing all grid detail files")

    # Calculate total size (need to use rglob for nested structure)
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
    Main function to generate compact two-tier manifest.
    """
    print("=" * 80)
    print("Compact Parquet Partition Manifest Generation")
    print("=" * 80)

    # Define output directory
    base_path = Path(config["dir"]["base"])
    version = config["dataset"]["version"]
    output_dir = base_path / "manifests" / f"v{version}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nOutput directory: {output_dir}")

    # Generate compact manifest
    manifest = generate_compact_manifest(config, output_dir)

    print("\n" + "=" * 80)
    print("Manifest generation complete!")
    print("=" * 80)
    print("\nGenerated structure:")
    print(f"  {output_dir}/")
    print("    ├── manifest.json          (main index)")
    print(
        f"    └── grids/                 ({manifest['total_grids']} grid detail files)"
    )
    print("\nUsage:")
    print("  - Load manifest.json for fast grid-level queries")
    print("  - Lazy-load individual grid files as needed")
    print("  - Use TidalManifestQuery class for spatial queries")


if __name__ == "__main__":
    main()
