"""
Query tidal data by geographic point, line, or area.

This script provides a CLI for querying tidal parquet data by:
- Point: Single lat/lon coordinate (finds nearest data point)
- Line: Start/end coordinates (finds all points along path)
- Area: Bounding box (finds all points within region)

Uses a local cache (./us_tidal_cache/) with ETag-based validation to avoid
redundant downloads from S3.

Usage:
    # Point query (default) - find nearest point
    python query_point.py --lat 60.73 --lon -151.43

    # Line query - find all points along a path
    python query_point.py --mode line --start-lat 60.7 --start-lon -151.4 --end-lat 60.8 --end-lon -151.5

    # Area query - find all points in bounding box
    python query_point.py --mode area --lat-min 60.7 --lat-max 60.8 --lon-min -151.5 --lon-max -151.4

    # Use S3 staging bucket
    python query_point.py --lat 60.73 --lon -151.43 --s3-bucket oedi-data-drop --aws-profile us-tidal

    # Use HPC local filesystem
    python query_point.py --lat 60.73 --lon -151.43 --use-hpc

    # Fast cached query (skip S3 validation)
    python query_point.py --lat 60.73 --lon -151.43 --skip-validation
"""

import argparse
import re
import time
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from config import config


class Timer:
    """Simple context manager for timing code blocks."""

    def __init__(self, name: str, enabled: bool = True):
        self.name = name
        self.enabled = enabled
        self.elapsed = 0.0

    def __enter__(self):
        if self.enabled:
            self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        if self.enabled:
            self.elapsed = time.perf_counter() - self.start
            print(f"  [{self.name}] {self.elapsed:.3f}s")


def parse_semver(version_str: str) -> Tuple[int, int, int]:
    """Parse semantic version string into tuple for comparison."""
    match = re.match(r"(\d+)\.(\d+)\.(\d+)", version_str)
    if not match:
        raise ValueError(f"Invalid semver: {version_str}")
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def find_latest_manifest_hpc(base_path: str) -> Optional[Path]:
    """
    Find the latest manifest on HPC filesystem using semver traversal.

    Parameters
    ----------
    base_path : str
        HPC base path (e.g., /projects/hindcastra/Tidal/datasets/high_resolution_tidal_hindcast)

    Returns
    -------
    Path or None
        Path to latest manifest file, or None if not found
    """
    manifests_dir = Path(base_path) / "manifest"

    if not manifests_dir.exists():
        print(f"  Manifest directory not found: {manifests_dir}")
        return None

    # Find all version directories (v1.0.0, v1.0.1, etc.)
    version_dirs = []
    for d in manifests_dir.iterdir():
        if d.is_dir() and d.name.startswith("v"):
            try:
                version = parse_semver(d.name[1:])  # Remove 'v' prefix
                version_dirs.append((d, version))
            except ValueError:
                continue

    # Sort by version descending
    version_dirs.sort(key=lambda x: x[1], reverse=True)

    for version_dir, _ in version_dirs:
        # Find all manifest files in this version directory
        manifest_files = []
        for f in version_dir.glob("manifest_*.json"):
            match = re.search(r"manifest_(\d+\.\d+\.\d+)\.json", f.name)
            if match:
                try:
                    version = parse_semver(match.group(1))
                    manifest_files.append((f, version))
                except ValueError:
                    continue

        # Sort by version descending
        manifest_files.sort(key=lambda x: x[1], reverse=True)

        if manifest_files:
            manifest_file = manifest_files[0][0]
            print(f"  Found manifest: {manifest_file}")
            return manifest_file

    return None


def find_latest_manifest_s3(
    s3_cache, benchmark: bool = False, skip_validation: bool = False
) -> Optional[Tuple[Path, str]]:
    """
    Find the latest manifest on S3 using semver traversal.

    If skip_validation is True and a cached manifest exists, uses the cache
    without hitting S3 at all. Otherwise, lists top-level version directories
    and downloads/caches the latest manifest.

    Parameters
    ----------
    s3_cache : S3CacheManager
        S3 cache manager instance
    benchmark : bool
        If True, print timing for sub-operations
    skip_validation : bool
        If True, use cached manifest without checking S3

    Returns
    -------
    tuple of (Path, str) or None
        (Path to cached manifest file, version string), or None if not found
    """
    # If skip_validation, check local cache first to avoid S3 entirely
    if skip_validation:
        with Timer("Check local cache", benchmark):
            manifest_cache_dir = s3_cache.cache_dir / "manifest"
            if manifest_cache_dir.exists():
                # Find version directories in cache
                version_dirs = []
                for d in manifest_cache_dir.iterdir():
                    if d.is_dir() and d.name.startswith("v"):
                        try:
                            version = parse_semver(d.name[1:])
                            version_dirs.append((d, version))
                        except ValueError:
                            continue

                if version_dirs:
                    # Sort by version descending and get latest
                    version_dirs.sort(key=lambda x: x[1], reverse=True)
                    latest_dir, latest_version_tuple = version_dirs[0]
                    latest_version = ".".join(map(str, latest_version_tuple))

                    # Look for manifest file
                    manifest_file = latest_dir / f"manifest_{latest_version}.json"
                    if manifest_file.exists():
                        print(f"  Using cached manifest: {manifest_file}")
                        return manifest_file, latest_version

    # Fall through to S3 if no valid cache or skip_validation=False
    manifest_prefix = f"{s3_cache.prefix}/manifest/"

    try:
        with Timer("S3 list version directories", benchmark):
            # List only top-level directories under manifest/ using Delimiter
            # This returns CommonPrefixes for directories, not all objects
            response = s3_cache.s3.list_objects_v2(
                Bucket=s3_cache.bucket,
                Prefix=manifest_prefix,
                Delimiter="/",
            )

            # Extract version directories from CommonPrefixes
            version_dirs = []
            for prefix_info in response.get("CommonPrefixes", []):
                prefix = prefix_info["Prefix"]  # e.g., "us-tidal/manifest/v1.0.0/"
                # Extract version from directory name
                dir_name = prefix.rstrip("/").split("/")[-1]  # e.g., "v1.0.0"
                if dir_name.startswith("v"):
                    try:
                        version = parse_semver(dir_name[1:])
                        version_dirs.append((dir_name, version))
                    except ValueError:
                        continue

        if not version_dirs:
            print(
                f"  No manifest versions found in s3://{s3_cache.bucket}/{manifest_prefix}"
            )
            return None

        # Sort by version (descending) and get latest
        version_dirs.sort(key=lambda x: x[1], reverse=True)
        latest_version_dir = version_dirs[0][0]  # e.g., "v1.0.0"
        latest_version = ".".join(map(str, version_dirs[0][1]))  # e.g., "1.0.0"

        # Construct manifest path directly
        manifest_key = f"{manifest_prefix}{latest_version_dir}/manifest_{latest_version}.json"
        print(f"  Found latest manifest: s3://{s3_cache.bucket}/{manifest_key}")

        # Get relative path (strip prefix)
        relative_path = manifest_key[len(s3_cache.prefix) + 1 :]  # +1 for the '/'

        # Download/cache the manifest
        with Timer("Cache get (manifest)", benchmark):
            local_path = s3_cache.get(relative_path, validate=not skip_validation)
        print(f"  Cached at: {local_path}")

        return local_path, latest_version

    except Exception as e:
        print(f"  Error accessing S3: {e}")
        return None


def handle_point_query(args, query, s3_cache, benchmark, total_start):
    """Handle single point query."""
    print("\nSearching for nearest point...")
    with Timer("query_nearest_point", benchmark):
        result = query.query_nearest_point(
            lat=args.lat,
            lon=args.lon,
            load_details=False,
        )

    if result is None:
        print("\nNo data points found near the query location.")
        return 1

    # Check distance threshold
    if args.max_distance_km and result["distance_km"] > args.max_distance_km:
        print(
            f"\nNearest point is {result['distance_km']:.2f} km away, "
            f"exceeds max distance of {args.max_distance_km} km"
        )
        return 1

    # Display result
    print("\n" + "-" * 70)
    print("NEAREST POINT FOUND")
    print("-" * 70)
    print(f"  Face ID:      {result['point']['face_id']}")
    print(f"  Latitude:     {result['point']['lat']}")
    print(f"  Longitude:    {result['point']['lon']}")
    print(f"  Distance:     {result['distance_km']:.4f} km from query point")
    print(f"  Location:     {result['location']}")
    print(f"  Grid ID:      {result['grid_id']}")

    # Get version info
    version_info = query.get_location_version_info(result["location"])
    data_version = args.data_version or version_info["latest_version"]
    print(f"  Data Version: {data_version}")

    # Get file path
    relative_path = result["point"]["file_path"]
    print(f"\n  Relative path: {relative_path}")

    if args.use_hpc:
        full_path = query.get_hpc_path(relative_path)
        print(f"  HPC path:      {full_path}")
    else:
        s3_uri = query.get_s3_uri(relative_path)
        print(f"  S3 URI:        {s3_uri}")

    if args.info_only:
        print("\n(--info-only specified, skipping parquet data load)")
        if benchmark:
            total_elapsed = time.perf_counter() - total_start
            print(f"\n  [TOTAL] {total_elapsed:.3f}s")
        return 0

    # Load and display parquet data
    return load_and_display_parquet(
        args, s3_cache, relative_path, benchmark, total_start
    )


def handle_line_query(args, query, s3_cache, benchmark, total_start):
    """Handle line query - find all points along a path."""
    print("\nSearching for points along line...")
    with Timer("query_all_on_line", benchmark):
        results = query.query_all_on_line(
            start_lat=args.start_lat,
            start_lon=args.start_lon,
            end_lat=args.end_lat,
            end_lon=args.end_lon,
            max_distance_deg=args.max_distance_from_line,
            load_details=True,
        )

    if not results:
        print("\nNo data points found along the line.")
        return 1

    # Display results summary
    print("\n" + "-" * 70)
    print(f"POINTS FOUND ALONG LINE: {len(results)}")
    print("-" * 70)

    # Build summary table
    print(
        f"\n{'#':<4} {'Face ID':<10} {'Latitude':<12} {'Longitude':<13} "
        f"{'Dist from line':<15} {'Dist along line':<15} {'Location'}"
    )
    print("-" * 95)

    for i, r in enumerate(results[:args.head]):
        # Get first point from grid details
        if "details" in r and "points" in r["details"]:
            points = r["details"]["points"]
            cols = r["details"].get("points_columns", ["lat", "lon", "face_id"])
            lat_idx = cols.index("lat") if "lat" in cols else 0
            lon_idx = cols.index("lon") if "lon" in cols else 1
            face_idx = cols.index("face_id") if "face_id" in cols else 2

            # Just show first point from grid
            pt = points[0]
            face_id = pt[face_idx]
            lat = pt[lat_idx]
            lon = pt[lon_idx]
        else:
            face_id = "N/A"
            lat, lon = r["centroid"]

        dist_from = r.get("distance_from_line_deg", 0)
        dist_along = r.get("distance_along_line_deg", 0)
        location = r.get("location", "unknown")

        print(
            f"{i+1:<4} {face_id:<10} {lat:<12} {lon:<13} "
            f"{dist_from:<15.4f} {dist_along:<15.4f} {location}"
        )

    if len(results) > args.head:
        print(f"\n... and {len(results) - args.head} more points (use --head to show more)")

    # Show unique locations
    locations = set()
    for r in results:
        if "location" in r:
            locations.add(r["location"])
    print(f"\nLocations: {', '.join(sorted(locations))}")

    if benchmark:
        total_elapsed = time.perf_counter() - total_start
        print(f"\n  [TOTAL] {total_elapsed:.3f}s")

    if s3_cache:
        stats = s3_cache.cache_stats()
        print(f"\nCache: {stats['total_files']} files, {stats['total_size_mb']:.2f} MB")

    return 0


def handle_area_query(args, query, s3_cache, benchmark, total_start):
    """Handle area query - find all points in bounding box."""
    print("\nSearching for points in bounding box...")
    with Timer("query_all_within_rectangular_area", benchmark):
        results = query.query_all_within_rectangular_area(
            lat_min=args.lat_min,
            lat_max=args.lat_max,
            lon_min=args.lon_min,
            lon_max=args.lon_max,
            load_details=True,
        )

    if not results:
        print("\nNo data points found in the bounding box.")
        return 1

    # Display results summary
    print("\n" + "-" * 70)
    print(f"POINTS FOUND IN AREA: {len(results)}")
    print("-" * 70)

    # Count total points across all grids
    total_points = 0
    for r in results:
        if "details" in r and "points" in r["details"]:
            total_points += len(r["details"]["points"])

    print(f"Total grids: {len(results)}")
    print(f"Total data points: {total_points}")

    # Build summary table
    print(
        f"\n{'#':<4} {'Grid ID':<20} {'Centroid Lat':<13} {'Centroid Lon':<13} "
        f"{'Points':<8} {'Location'}"
    )
    print("-" * 80)

    for i, r in enumerate(results[:args.head]):
        grid_id = r.get("grid_id", "N/A")
        cent_lat, cent_lon = r["centroid"]
        n_points = len(r["details"]["points"]) if "details" in r and "points" in r["details"] else 0
        location = r.get("location", "unknown")

        print(
            f"{i+1:<4} {grid_id:<20} {cent_lat:<13.7f} {cent_lon:<13.7f} "
            f"{n_points:<8} {location}"
        )

    if len(results) > args.head:
        print(f"\n... and {len(results) - args.head} more grids (use --head to show more)")

    # Show unique locations
    locations = set()
    for r in results:
        if "location" in r:
            locations.add(r["location"])
    print(f"\nLocations: {', '.join(sorted(locations))}")

    if benchmark:
        total_elapsed = time.perf_counter() - total_start
        print(f"\n  [TOTAL] {total_elapsed:.3f}s")

    if s3_cache:
        stats = s3_cache.cache_stats()
        print(f"\nCache: {stats['total_files']} files, {stats['total_size_mb']:.2f} MB")

    return 0


def load_and_display_parquet(args, s3_cache, relative_path, benchmark, total_start):
    """Load and display parquet data."""
    print("\n" + "-" * 70)
    print("PARQUET DATA")
    print("-" * 70)

    try:
        if args.use_hpc:
            from query_tidal_manifest import TidalManifestQuery

            full_path = TidalManifestQuery.get_hpc_path(relative_path)
            parquet_path = Path(full_path)
            if not parquet_path.exists():
                print(f"\nERROR: File not found: {parquet_path}")
                return 1
        else:
            # Download from S3 using cache
            print("\n  Fetching parquet file...")
            parquet_path = s3_cache.get(relative_path, validate=False)

        print(f"  Loading: {parquet_path}")
        df = pd.read_parquet(parquet_path)

        print(f"\nShape: {df.shape[0]} rows x {df.shape[1]} columns")
        print(f"Columns: {list(df.columns)}")
        print(f"\nFirst {args.head} rows:")
        print(df.head(args.head).to_string())

        # Show basic stats
        print("\nData summary:")
        numeric_cols = df.select_dtypes(include=["number"]).columns
        if len(numeric_cols) > 0:
            print(df[numeric_cols].describe().to_string())

    except Exception as e:
        print(f"\nERROR loading parquet: {e}")
        import traceback

        traceback.print_exc()
        return 1

    print("\n" + "=" * 70)
    print("Query complete!")
    print("=" * 70)

    # Show final cache stats
    if s3_cache:
        stats = s3_cache.cache_stats()
        print(f"Cache: {stats['total_files']} files, {stats['total_size_mb']:.2f} MB")

    if benchmark:
        total_elapsed = time.perf_counter() - total_start
        print(f"\n  [TOTAL] {total_elapsed:.3f}s")

    return 0


def main():
    # Get defaults from config
    storage_config = config["storage"]
    default_hpc_base = storage_config["hpc_base_path"]
    default_s3_bucket = storage_config["s3_bucket"]
    default_s3_prefix = storage_config["s3_prefix"]

    parser = argparse.ArgumentParser(
        description="Query tidal data by geographic point, line, or area",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Point query (default) - find nearest point
    python query_point.py --lat 60.73 --lon -151.43

    # Line query - find all points along a path
    python query_point.py --mode line --start-lat 60.7 --start-lon -151.4 \\
                          --end-lat 60.8 --end-lon -151.5

    # Area query - find all points in bounding box
    python query_point.py --mode area --lat-min 60.7 --lat-max 60.8 \\
                          --lon-min -151.5 --lon-max -151.4

    # Use S3 staging bucket
    python query_point.py --lat 60.73 --lon -151.43 \\
                          --s3-bucket oedi-data-drop --aws-profile us-tidal

    # Fast cached query (skip S3 validation)
    python query_point.py --lat 60.73 --lon -151.43 --skip-validation

Test coordinates:
    Cook Inlet:        --lat 60.7320786 --lon -151.4315796
    Piscataqua River:  --lat 43.0521126 --lon -70.7007828
        """,
    )

    # Query mode
    parser.add_argument(
        "--mode",
        type=str,
        choices=["point", "line", "area"],
        default="point",
        help="Query mode: point (single location), line (path), or area (bounding box)",
    )

    # Point query arguments
    parser.add_argument(
        "--lat", type=float, help="Query latitude for point mode (decimal degrees)"
    )
    parser.add_argument(
        "--lon", type=float, help="Query longitude for point mode (decimal degrees)"
    )

    # Line query arguments
    parser.add_argument(
        "--start-lat", type=float, help="Starting latitude for line mode"
    )
    parser.add_argument(
        "--start-lon", type=float, help="Starting longitude for line mode"
    )
    parser.add_argument("--end-lat", type=float, help="Ending latitude for line mode")
    parser.add_argument("--end-lon", type=float, help="Ending longitude for line mode")
    parser.add_argument(
        "--max-distance-from-line",
        type=float,
        default=0.1,
        help="Max perpendicular distance from line in degrees (default: 0.1)",
    )

    # Area query arguments
    parser.add_argument("--lat-min", type=float, help="Minimum latitude for area mode")
    parser.add_argument("--lat-max", type=float, help="Maximum latitude for area mode")
    parser.add_argument("--lon-min", type=float, help="Minimum longitude for area mode")
    parser.add_argument("--lon-max", type=float, help="Maximum longitude for area mode")

    # Data source options
    parser.add_argument(
        "--use-hpc",
        action="store_true",
        help="Use HPC local filesystem instead of S3",
    )
    parser.add_argument(
        "--hpc-base-path",
        type=str,
        default=default_hpc_base,
        help=f"HPC base path (default: {default_hpc_base})",
    )

    # AWS options
    parser.add_argument(
        "--aws-profile",
        type=str,
        default=None,
        help="AWS profile name for S3 access",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="./us_tidal_cache",
        help="Local cache directory (default: ./us_tidal_cache)",
    )

    # S3 configuration
    parser.add_argument(
        "--s3-bucket",
        type=str,
        default=default_s3_bucket,
        help=f"S3 bucket name (default: {default_s3_bucket})",
    )
    parser.add_argument(
        "--s3-prefix",
        type=str,
        default=default_s3_prefix,
        help=f"S3 prefix (default: {default_s3_prefix})",
    )

    # Version options
    parser.add_argument(
        "--manifest-version",
        type=str,
        default=None,
        help="Specific manifest version to use (default: latest)",
    )
    parser.add_argument(
        "--data-version",
        type=str,
        default=None,
        help="Specific data version to use (default: latest for location)",
    )

    # Output options
    parser.add_argument(
        "--head",
        type=int,
        default=10,
        help="Number of rows to display (default: %(default)s)",
    )
    parser.add_argument(
        "--max-distance-km",
        type=float,
        default=None,
        help="Maximum distance in km to accept (default: no limit)",
    )
    parser.add_argument(
        "--info-only",
        action="store_true",
        help="Only show point info, don't load parquet data",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear local cache before running",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Show timing benchmarks for each step",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip S3 ETag validation (use cached files without checking if stale)",
    )

    args = parser.parse_args()
    benchmark = args.benchmark
    skip_validation = args.skip_validation

    # Validate arguments based on mode
    if args.mode == "point":
        if args.lat is None or args.lon is None:
            parser.error("Point mode requires --lat and --lon")
    elif args.mode == "line":
        if any(
            v is None
            for v in [args.start_lat, args.start_lon, args.end_lat, args.end_lon]
        ):
            parser.error(
                "Line mode requires --start-lat, --start-lon, --end-lat, --end-lon"
            )
    elif args.mode == "area":
        if any(
            v is None
            for v in [args.lat_min, args.lat_max, args.lon_min, args.lon_max]
        ):
            parser.error(
                "Area mode requires --lat-min, --lat-max, --lon-min, --lon-max"
            )

    total_start = time.perf_counter()

    print("=" * 70)
    print("Tidal Data Query")
    print("=" * 70)

    # Print query info based on mode
    if args.mode == "point":
        print(f"\nMode: POINT")
        print(f"Query coordinates: ({args.lat}, {args.lon})")
    elif args.mode == "line":
        print(f"\nMode: LINE")
        print(f"Start: ({args.start_lat}, {args.start_lon})")
        print(f"End:   ({args.end_lat}, {args.end_lon})")
        print(f"Max distance from line: {args.max_distance_from_line}°")
    elif args.mode == "area":
        print(f"\nMode: AREA (bounding box)")
        print(f"Latitude:  {args.lat_min}° to {args.lat_max}°")
        print(f"Longitude: {args.lon_min}° to {args.lon_max}°")

    # Initialize S3 cache if not using HPC
    s3_cache = None
    if not args.use_hpc:
        with Timer("Import S3CacheManager", benchmark):
            from s3_cache_manager import S3CacheManager

        with Timer("Initialize S3CacheManager", benchmark):
            # Include bucket name in cache directory to separate different buckets
            cache_dir = Path(args.cache_dir) / args.s3_bucket
            s3_cache = S3CacheManager(
                bucket=args.s3_bucket,
                prefix=args.s3_prefix,
                cache_dir=cache_dir,
                aws_profile=args.aws_profile,
            )

        if args.clear_cache:
            print(f"\nClearing cache: {cache_dir}")
            s3_cache.clear_cache()

        # Show cache stats
        stats = s3_cache.cache_stats()
        print(
            f"\nCache: {stats['cache_dir']} ({stats['total_files']} files, {stats['total_size_mb']:.2f} MB)"
        )

    # Find manifest
    print("\nLocating manifest...")
    if args.use_hpc:
        print(f"  Source: HPC filesystem ({args.hpc_base_path})")
        with Timer("Find manifest (HPC)", benchmark):
            manifest_path = find_latest_manifest_hpc(args.hpc_base_path)
        if manifest_path is None:
            print("\nERROR: Could not find manifest")
            return 1
    else:
        print(f"  Source: S3 (s3://{args.s3_bucket}/{args.s3_prefix})")
        with Timer("Find manifest (S3)", benchmark):
            result = find_latest_manifest_s3(s3_cache, benchmark, skip_validation)
        if result is None:
            print("\nERROR: Could not find manifest")
            return 1
        manifest_path, manifest_version = result

    # Load manifest and create query interface
    print("\nLoading manifest...")
    with Timer("Import TidalManifestQuery", benchmark):
        from query_tidal_manifest import TidalManifestQuery

    # Pass s3_cache to TidalManifestQuery for on-demand grid file fetching
    with Timer("Initialize TidalManifestQuery (load JSON + build KDTree)", benchmark):
        query = TidalManifestQuery(manifest_path, s3_cache=s3_cache)

    # Execute query based on mode
    if args.mode == "point":
        return handle_point_query(args, query, s3_cache, benchmark, total_start)
    elif args.mode == "line":
        return handle_line_query(args, query, s3_cache, benchmark, total_start)
    elif args.mode == "area":
        return handle_area_query(args, query, s3_cache, benchmark, total_start)

    return 0


if __name__ == "__main__":
    exit(main())
