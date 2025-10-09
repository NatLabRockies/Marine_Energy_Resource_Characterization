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
    # Split by '.' to get components
    parts = filename.split('.')

    # Minimum structure: location.dataset.face=X.lat=Y.lon=Z-temporal.b4.date.time.parquet
    # That's at least 9 parts
    if len(parts) < 9:
        raise ValueError(f"Filename has too few components (expected >= 9, got {len(parts)}): {filename}")

    # Extract components by position and known patterns
    location = parts[0]
    dataset = parts[1]

    # Find face= component
    face_part = None
    lat_part = None
    lon_temporal_part = None

    for i, part in enumerate(parts):
        if part.startswith('face='):
            face_part = part
        elif part.startswith('lat='):
            lat_part = part
        elif part.startswith('lon='):
            # This part contains lon and temporal: lon={value}-{temporal}
            lon_temporal_part = part

    if not face_part or not lat_part or not lon_temporal_part:
        raise ValueError(f"Missing required components (face=, lat=, lon=) in filename: {filename}")

    # Parse face ID
    face_id = int(face_part.replace('face=', ''))

    # Parse latitude
    lat = float(lat_part.replace('lat=', ''))

    # Parse longitude and temporal (format: lon={value}-{temporal})
    # The temporal is separated by hyphen
    lon_and_temporal = lon_temporal_part.replace('lon=', '')

    # Find the last hyphen which separates lon from temporal
    # Need to handle negative longitudes like -174.9613647-1h
    last_hyphen_idx = lon_and_temporal.rfind('-')
    if last_hyphen_idx == -1:
        raise ValueError(f"Could not find temporal separator in lon component: {lon_temporal_part}")

    # For negative longitudes, we need to find the hyphen that's NOT at position 0
    if lon_and_temporal[0] == '-':
        # Negative longitude, find the second hyphen
        second_hyphen = lon_and_temporal.find('-', 1)
        if second_hyphen == -1:
            raise ValueError(f"Could not find temporal separator for negative longitude: {lon_temporal_part}")
        lon = float(lon_and_temporal[:second_hyphen])
        temporal = lon_and_temporal[second_hyphen + 1:]
    else:
        # Positive longitude, use last hyphen
        lon = float(lon_and_temporal[:last_hyphen_idx])
        temporal = lon_and_temporal[last_hyphen_idx + 1:]

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


def scan_parquet_partitions(partition_dir):
    """
    Scan parquet partition directory and collect metadata for all files.

    Parameters
    ----------
    partition_dir : Path
        Root directory containing parquet partitions

    Returns
    -------
    list
        List of dictionaries containing file metadata
    """
    partition_dir = Path(partition_dir)

    if not partition_dir.exists():
        raise ValueError(f"Partition directory does not exist: {partition_dir}")

    print(f"Scanning directory: {partition_dir}")

    # Find all parquet files
    parquet_files = sorted(partition_dir.rglob("*.parquet"))
    print(f"Found {len(parquet_files)} parquet files")

    file_metadata = []

    for idx, file_path in enumerate(parquet_files):
        if idx % 10000 == 0 and idx > 0:
            print(f"  Processed {idx}/{len(parquet_files)} files...")

        # Parse filename - will raise ValueError on failure
        try:
            file_info = parse_parquet_filename(file_path.name)
        except ValueError as e:
            print(f"\nERROR: Could not parse filename: {file_path.name}")
            print(f"Full path: {file_path}")
            print(f"Parse error: {e}")
            raise

        # Parse partition path
        relative_path = file_path.relative_to(partition_dir)
        partition_info = parse_partition_path(relative_path.parent)
        if partition_info is None:
            print(f"\nERROR: Could not parse partition path: {relative_path.parent}")
            print(f"Full path: {file_path}")
            raise ValueError(f"Failed to parse partition path: {relative_path.parent}")

        # Get file size
        file_size = file_path.stat().st_size

        # Combine metadata
        metadata = {
            "face_id": file_info["face_id"],
            "lat": file_info["lat"],
            "lon": file_info["lon"],
            "location": file_info["location"],
            "dataset": file_info["dataset"],
            "temporal": file_info["temporal"],
            "partition": partition_info,
            "file_path": str(relative_path),
            "file_size": file_size,
        }

        file_metadata.append(metadata)

    print(f"Successfully processed {len(file_metadata)} files")
    return file_metadata


def build_spatial_index(file_metadata):
    """
    Build spatial grid index for fast lat/lon lookups.

    The index is organized by partition keys for O(1) lookup time.

    Parameters
    ----------
    file_metadata : list
        List of file metadata dictionaries

    Returns
    -------
    dict
        Spatial index mapping partition keys to file lists
    """
    spatial_index = {}

    for metadata in file_metadata:
        partition = metadata["partition"]

        # Create partition key
        key = f"lat_deg={partition['lat_deg']}/lon_deg={partition['lon_deg']}/lat_dec={partition['lat_dec']}/lon_dec={partition['lon_dec']}"

        if key not in spatial_index:
            spatial_index[key] = []

        spatial_index[key].append({
            "face_id": metadata["face_id"],
            "lat": metadata["lat"],
            "lon": metadata["lon"],
            "file_path": metadata["file_path"],
        })

    return spatial_index


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


def generate_combined_manifest(config, output_path):
    """
    Generate combined manifest with spatial and face ID indices.

    Parameters
    ----------
    config : dict
        Configuration dictionary
    output_path : Path
        Output path for the combined manifest JSON
    """
    print("\n=== Generating Combined Manifest ===")

    # Collect file metadata from all locations
    all_file_metadata = []
    location_stats = {}

    for location_key, location in config["location_specification"].items():
        print(f"\nProcessing location: {location['label']} ({location_key})")

        partition_dir = file_manager.get_vap_partition_output_dir(config, location)

        if not partition_dir.exists():
            print(f"  Partition directory does not exist, skipping: {partition_dir}")
            continue

        file_metadata = scan_parquet_partitions(partition_dir)

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

    print(f"\nTotal files across all locations: {len(all_file_metadata)}")

    # Build indices
    print("\nBuilding spatial index...")
    spatial_index = build_spatial_index(all_file_metadata)
    print(f"  Created {len(spatial_index)} spatial grid cells")

    print("\nBuilding face ID index...")
    faceid_index = build_faceid_index(all_file_metadata)
    print(f"  Created indices for {len(faceid_index)} locations")

    # Compute bounds
    print("\nComputing spatial bounds...")
    bounds = compute_bounds(all_file_metadata)

    # Create combined manifest
    manifest = {
        "manifest_version": config["dataset"]["version"],
        "metadata": {
            "dataset_name": config["dataset"]["name"],
            "dataset_label": config["dataset"]["label"],
            "dataset_version": config["dataset"]["version"],
            "manifest_generated": datetime.now().isoformat(),
            "total_points": len(all_file_metadata),
            "total_grid_cells": len(spatial_index),
            "partition_decimal_places": config["partition"]["decimal_places"],
            "locations": location_stats,
        },
        "spatial_bounds": bounds,
        "spatial_index": spatial_index,
        "faceid_index": faceid_index,
    }

    # Write manifest
    print(f"\nWriting combined manifest to: {output_path}")
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  Manifest size: {file_size_mb:.2f} MB")

    return manifest


def generate_spatial_manifest(manifest, output_path):
    """
    Generate spatial-only manifest (grid index without face IDs).

    Parameters
    ----------
    manifest : dict
        Combined manifest dictionary
    output_path : Path
        Output path for the spatial manifest JSON
    """
    print("\n=== Generating Spatial-Only Manifest ===")

    spatial_manifest = {
        "metadata": manifest["metadata"],
        "spatial_bounds": manifest["spatial_bounds"],
        "spatial_index": manifest["spatial_index"],
    }

    print(f"Writing spatial manifest to: {output_path}")
    with open(output_path, "w") as f:
        json.dump(spatial_manifest, f, indent=2)

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  Manifest size: {file_size_mb:.2f} MB")


def generate_faceid_manifest(manifest, output_path):
    """
    Generate face ID-only manifest (direct face lookup without spatial index).

    Parameters
    ----------
    manifest : dict
        Combined manifest dictionary
    output_path : Path
        Output path for the face ID manifest JSON
    """
    print("\n=== Generating Face ID-Only Manifest ===")

    faceid_manifest = {
        "metadata": manifest["metadata"],
        "faceid_index": manifest["faceid_index"],
    }

    print(f"Writing face ID manifest to: {output_path}")
    with open(output_path, "w") as f:
        json.dump(faceid_manifest, f, indent=2)

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  Manifest size: {file_size_mb:.2f} MB")


def main():
    """
    Main function to generate all manifest files.
    """
    print("=" * 80)
    print("Parquet Partition Manifest Generation")
    print("=" * 80)

    # Define output directory (same as base or separate manifest directory)
    base_path = Path(config["dir"]["base"])
    output_dir = base_path / "manifests"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nOutput directory: {output_dir}")

    # Get version for manifest filenames
    version = config["dataset"]["version"]

    # Generate combined manifest
    combined_manifest_path = output_dir / f"high_res_tidal_point_manifest.v{version}.json"
    manifest = generate_combined_manifest(config, combined_manifest_path)

    # Generate spatial-only manifest
    spatial_manifest_path = output_dir / f"high_res_tidal_spatial_manifest.v{version}.json"
    generate_spatial_manifest(manifest, spatial_manifest_path)

    # Generate face ID-only manifest
    faceid_manifest_path = output_dir / f"high_res_tidal_faceid_manifest.v{version}.json"
    generate_faceid_manifest(manifest, faceid_manifest_path)

    print("\n" + "=" * 80)
    print("Manifest generation complete!")
    print("=" * 80)
    print(f"\nGenerated files:")
    print(f"  1. {combined_manifest_path}")
    print(f"  2. {spatial_manifest_path}")
    print(f"  3. {faceid_manifest_path}")
    print("\nThese manifests can be used for efficient spatial queries:")
    print("  - Point queries: Find nearest parquet file for a given lat/lon")
    print("  - Polygon queries: Find all files within a bounding box")
    print("  - Face ID queries: Direct lookup by face identifier")
    print("  - Compatible with both Python and JavaScript (JSON format)")


if __name__ == "__main__":
    main()
