#!/usr/bin/env python3
"""
Create SQLite manifest for S3 upload tracking.

Discovers all files for a location/data_level, filters by extension,
and creates a SQLite database for tracking upload progress.
"""

import argparse
import subprocess
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

from config import config
from src.file_manager import (
    get_manifest_output_dir,
    get_output_dirs,
    validate_manifest_version,
)

# Data level file type and recursion configuration
DATA_LEVEL_FILE_CONFIG = {
    "00_raw": {"valid_extensions": [".nc"], "should_recurse": True},
    "a1_std": {"valid_extensions": [".nc"], "should_recurse": False},
    "a2_std_partition": {"valid_extensions": [".nc"], "should_recurse": False},
    "b1_vap": {"valid_extensions": [".nc"], "should_recurse": False},
    "b1_vap_daily_compressed": {"valid_extensions": [".nc"], "should_recurse": False},
    "b2_monthly_mean_vap": {"valid_extensions": [".nc"], "should_recurse": False},
    "b3_yearly_mean_vap": {"valid_extensions": [".nc"], "should_recurse": False},
    "b1_vap_by_point_partition": {
        "valid_extensions": [".parquet"],
        "should_recurse": True,
    },
    "b4_vap_summary_parquet": {
        "valid_extensions": None,
        "should_recurse": True,
    },  # All files
    "b5_vap_atlas_summary_parquet": {
        "valid_extensions": None,
        "should_recurse": True,
    },  # All files
    "hsds": {"valid_extensions": [".h5"], "should_recurse": False},
    "manifest": {
        "valid_extensions": [".json"],
        "should_recurse": True,
        "is_global": True,  # Not location-specific
    },
}

# Location mapping from config
LOCATION_MAP = {
    key: spec["output_name"] for key, spec in config["location_specification"].items()
}


def create_manifest_table(conn):
    """Create the files table in SQLite database with WAL mode for better concurrency."""
    cursor = conn.cursor()

    # Enable WAL mode for better concurrent access
    cursor.execute("PRAGMA journal_mode=WAL")

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS files (
            file_index INTEGER PRIMARY KEY,
            local_path TEXT NOT NULL,
            relative_path TEXT NOT NULL,
            s3_destination TEXT NOT NULL,
            file_size_bytes INTEGER NOT NULL,
            file_extension TEXT NOT NULL,
            upload_status TEXT DEFAULT 'pending',
            etag TEXT,
            upload_timestamp DATETIME
        )
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_upload_status ON files(upload_status)
    """)
    conn.commit()


def discover_manifest_files():
    """
    Discover manifest JSON files for upload.

    Validates that config manifest version matches filesystem latest version,
    then discovers all JSON files recursively.

    Returns:
        List of (local_path, relative_path, s3_destination, file_size, extension) tuples

    Raises:
        SystemExit: If manifest version mismatch between config and filesystem
    """
    # Validate manifest version consistency
    try:
        _, config_version, fs_version, manifest_dir = validate_manifest_version(config)
        print(
            f"Manifest version validated: {config_version} (config) = {fs_version} (filesystem)"
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Get file discovery configuration
    file_config = DATA_LEVEL_FILE_CONFIG["manifest"]
    valid_extensions = file_config["valid_extensions"]

    # S3 destination base path: manifest/v{manifest_version}/
    s3_base_relative_path = Path("manifest") / f"v{config_version}"

    # Discover files recursively
    print(f"Discovering files in: {manifest_dir}")
    print(f"  Extensions: {valid_extensions if valid_extensions else 'all'}")
    print("  Recursion: enabled")

    file_records = []
    for file_path in manifest_dir.rglob("*"):
        if file_path.is_file():
            if valid_extensions is None or file_path.suffix in valid_extensions:
                relative_path = file_path.relative_to(manifest_dir)
                s3_destination = str(s3_base_relative_path / relative_path)
                file_size = file_path.stat().st_size
                extension = file_path.suffix

                file_records.append(
                    (
                        str(file_path),
                        str(relative_path),
                        s3_destination,
                        file_size,
                        extension,
                    )
                )

    # Sort for deterministic ordering
    file_records.sort(key=lambda x: x[0])

    print(f"Found {len(file_records)} files")

    return file_records


def discover_files(location_key, data_level):
    """
    Discover all valid files for upload using versioned directories from file_manager.

    Args:
        location_key: Location key (e.g., 'cook_inlet') - ignored for global data levels
        data_level: Data level directory name (e.g., 'b1_vap', 'manifest')

    Returns:
        List of (local_path, relative_path, s3_destination, file_size, extension) tuples
    """
    # Validate data level
    if data_level not in DATA_LEVEL_FILE_CONFIG:
        print(f"Error: Unknown data level '{data_level}'", file=sys.stderr)
        print(
            f"Available data levels: {', '.join(DATA_LEVEL_FILE_CONFIG.keys())}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Handle global data levels (like manifest)
    file_config = DATA_LEVEL_FILE_CONFIG[data_level]
    if file_config.get("is_global", False):
        if data_level == "manifest":
            return discover_manifest_files()
        else:
            print(f"Error: Unknown global data level '{data_level}'", file=sys.stderr)
            sys.exit(1)

    # Get file discovery configuration for location-specific levels
    valid_extensions = file_config["valid_extensions"]
    should_recurse = file_config["should_recurse"]

    # Get location specification
    location_spec = config["location_specification"][location_key]

    # Get versioned output directories from file_manager (both full paths and relative paths)
    output_dirs_full = get_output_dirs(config, location_spec, omit_base_path=False)
    output_dirs_relative = get_output_dirs(config, location_spec, omit_base_path=True)

    # Map data_level to output_dir key
    # Note: 00_raw uses the input directory structure, not output
    data_level_to_dir_key = {
        "a1_std": "standardized",
        "a2_std_partition": "standardized_partition",
        "b1_vap": "vap",
        "b1_vap_daily_compressed": "vap_daily_compressed",
        "b2_monthly_mean_vap": "monthly_summary_vap",
        "b3_yearly_mean_vap": "yearly_summary_vap",
        "b1_vap_by_point_partition": "vap_partition",
        "b4_vap_summary_parquet": "vap_summary_parquet",
        "b5_vap_atlas_summary_parquet": "vap_atlas_summary_parquet",
        "hsds": "hsds",
    }

    # Handle 00_raw separately (uses input directory structure)
    if data_level == "00_raw":
        base_path = Path(config["dir"]["base"])
        input_dir_template = config["dir"]["input"]["original"]
        input_dir = input_dir_template.replace(
            "<location>", location_spec["output_name"]
        )
        local_dir = base_path / input_dir
        # For 00_raw, relative path for S3 is manually constructed
        s3_base_relative_path = (
            Path(location_spec["output_name"])
            / f"v{config['dataset']['version']}"
            / data_level
        )
    else:
        dir_key = data_level_to_dir_key.get(data_level)
        if not dir_key:
            print(
                f"Error: No directory mapping for data level '{data_level}'",
                file=sys.stderr,
            )
            sys.exit(1)

        local_dir = output_dirs_full[dir_key]
        # Get the relative path from file_manager for S3 construction (single source of truth)
        s3_base_relative_path = output_dirs_relative[dir_key]

    if not local_dir.exists():
        print(f"Error: Directory does not exist: {local_dir}", file=sys.stderr)
        return []

    # Discover files using find command (much faster than Python glob on Lustre)
    print(f"Discovering files in: {local_dir}")
    print(f"  Extensions: {valid_extensions if valid_extensions else 'all'}")
    print(f"  Recursion: {'enabled' if should_recurse else 'disabled'}")
    print("  Using 'find' command for fast discovery on Lustre...")

    # Build find command
    # Use -printf to get file size and path in one pass (avoids separate stat calls)
    # Format: "size<TAB>path<NEWLINE>"
    cmd = ["find", str(local_dir)]

    if not should_recurse:
        cmd.extend(["-maxdepth", "1"])

    cmd.extend(["-type", "f"])

    # Add extension filter using -name patterns
    if valid_extensions:
        if len(valid_extensions) == 1:
            cmd.extend(["-name", f"*{valid_extensions[0]}"])
        else:
            # Multiple extensions: ( -name "*.ext1" -o -name "*.ext2" )
            cmd.append("(")
            for i, ext in enumerate(valid_extensions):
                if i > 0:
                    cmd.append("-o")
                cmd.extend(["-name", f"*{ext}"])
            cmd.append(")")

    # Use -printf to get size and path in single pass (GNU find feature, available on Linux)
    cmd.extend(["-printf", "%s\\t%p\\n"])

    # Execute find command
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Warning: find command returned non-zero exit code: {result.returncode}")
        if result.stderr:
            print(f"  stderr: {result.stderr[:500]}")

    # Parse find output and build file records
    file_records = []
    lines = result.stdout.strip().split("\n") if result.stdout.strip() else []

    for line in lines:
        if not line or "\t" not in line:
            continue

        size_str, file_path_str = line.split("\t", 1)
        file_path = Path(file_path_str)

        try:
            file_size = int(size_str)
        except ValueError:
            # Fallback to stat if size parsing fails
            file_size = file_path.stat().st_size

        relative_path = file_path.relative_to(local_dir)
        s3_destination = str(s3_base_relative_path / relative_path)
        extension = file_path.suffix

        file_records.append(
            (str(file_path), str(relative_path), s3_destination, file_size, extension)
        )

    # Sort for deterministic ordering
    file_records.sort(key=lambda x: x[0])

    print(f"Found {len(file_records)} files")

    return file_records


def create_manifest(location_key, data_level=None, output_dir="cache/s3_upload"):
    """
    Create SQLite manifest for S3 uploads.

    Args:
        location_key: Location key (e.g., 'cook_inlet') or 'manifest' for global manifest
        data_level: Data level (ignored when location_key is 'manifest')
        output_dir: Directory to store manifest

    Returns:
        Path to created manifest file
    """
    # Handle special case: location_key == "manifest"
    is_manifest_upload = location_key == "manifest"

    if is_manifest_upload:
        # For manifest uploads, data_level is implicitly "manifest"
        data_level = "manifest"
    else:
        # Validate location for non-manifest uploads
        if location_key not in LOCATION_MAP:
            print(f"Error: Unknown location '{location_key}'", file=sys.stderr)
            print(
                f"Available locations: {', '.join(LOCATION_MAP.keys())}, manifest",
                file=sys.stderr,
            )
            sys.exit(1)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate manifest filename
    if is_manifest_upload:
        manifest_version = config["manifest"]["version"].replace(".", "_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        manifest_file = (
            output_path / f"manifest_global_v{manifest_version}_{timestamp}.db"
        )
    else:
        output_name = LOCATION_MAP[location_key]
        version = config["dataset"]["version"].replace(".", "_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        manifest_file = (
            output_path
            / f"manifest_{output_name}_{version}_{data_level}_{timestamp}.db"
        )

    print(f"\nCreating manifest: {manifest_file}")

    # Discover files
    file_records = discover_files(location_key, data_level)

    if not file_records:
        print("No files found to upload", file=sys.stderr)
        sys.exit(1)

    # Create SQLite database
    conn = sqlite3.connect(manifest_file)
    create_manifest_table(conn)

    # Insert file records
    cursor = conn.cursor()
    cursor.executemany(
        """
        INSERT INTO files (file_index, local_path, relative_path, s3_destination,
                          file_size_bytes, file_extension)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        [
            (idx, local_path, rel_path, s3_dest, size, ext)
            for idx, (local_path, rel_path, s3_dest, size, ext) in enumerate(
                file_records
            )
        ],
    )
    conn.commit()

    # Print summary
    total_size = sum(size for _, _, _, size, _ in file_records)
    print("\nManifest Summary:")
    print(f"  Total files:  {len(file_records)}")
    print(f"  Total size:   {total_size / (1024**3):.2f} GB")

    # Count by extension
    from collections import Counter

    ext_counts = Counter(ext for _, _, _, _, ext in file_records)
    print("\n  Files by type:")
    for ext, count in sorted(ext_counts.items()):
        print(f"    {ext}: {count}")

    conn.close()

    print(f"\nManifest created: {manifest_file}")
    return str(manifest_file)


def main():
    parser = argparse.ArgumentParser(
        description="Create SQLite manifest for S3 uploads"
    )

    parser.add_argument("location", help="Location key (e.g., cook_inlet)")

    parser.add_argument("data_level", help="Data level (e.g., b1_vap)")

    parser.add_argument(
        "--output-dir",
        default="cache/s3_upload",
        help="Directory to store manifest (default: cache/s3_upload)",
    )

    args = parser.parse_args()

    manifest_path = create_manifest(args.location, args.data_level, args.output_dir)
    print(f"\n{manifest_path}")


if __name__ == "__main__":
    main()
