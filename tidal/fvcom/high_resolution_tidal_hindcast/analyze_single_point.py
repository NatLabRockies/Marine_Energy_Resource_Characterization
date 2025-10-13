"""
Query and extract tidal data for a single point.
"""

from pathlib import Path

import pandas as pd

from query_tidal_manifest import TidalManifestQuery

# ============================================================================
# CONFIGURATION - Edit these coordinates
# ============================================================================
QUERY_LAT = 49.94
QUERY_LON = -174.96

# Manifest path (default: HPC location)
MANIFEST_PATH = Path(
    "/projects/hindcastra/Tidal/datasets/high_resolution_tidal_hindcast/manifests/v0.3.0/manifest.json"
)

# Base directory for datasets (location-specific subdirectories)
BASE_DATA_DIR = Path(
    "/projects/hindcastra/Tidal/datasets/high_resolution_tidal_hindcast"
)

# Output file path
OUTPUT_FILE = Path("./single_point_data.parquet")
# ============================================================================


def analyze_point(lat: float, lon: float):
    """
    Query tidal data for a single point and read the parquet file.

    Parameters
    ----------
    lat : float
        Latitude coordinate
    lon : float
        Longitude coordinate

    Returns
    -------
    pd.DataFrame
        Parquet data for the nearest point
    """
    # Check if manifest exists
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Manifest not found at {MANIFEST_PATH}\n"
            "Please update MANIFEST_PATH or run generate_parquet_partition_manifest_json.py first."
        )

    # Load manifest and build query interface
    print(f"Loading manifest from: {MANIFEST_PATH}")
    query = TidalManifestQuery(MANIFEST_PATH)

    # Query for nearest point
    print(f"\nQuerying point: ({lat}, {lon})")
    result = query.query_nearest_point(lat=lat, lon=lon, load_details=True)

    if result is None:
        raise ValueError("No grid found near the specified coordinates.")

    # Print query results
    print("\n" + "=" * 80)
    print("QUERY RESULTS")
    print("=" * 80)
    print(f"Grid ID: {result['grid_id']}")
    print(f"Grid Centroid: {result['centroid']}")
    print(
        f"Distance from query point: {result['distance_deg']:.6f}° (~{result['distance_deg'] * 111:.2f} km)"
    )

    details = result["details"]
    print(f"\nLocation: {details['location']}")
    print(f"Temporal Resolution: {details['temporal']}")
    print(f"Number of points in grid: {len(details['points'])}")

    # Find the closest point in the grid
    print("\nFinding closest point in grid...")
    min_distance = float("inf")
    closest_point = None

    for point in details["points"]:
        # Calculate distance from query point
        lat_diff = point["lat"] - lat
        lon_diff = point["lon"] - lon
        distance = (lat_diff**2 + lon_diff**2) ** 0.5

        if distance < min_distance:
            min_distance = distance
            closest_point = point

    print("\nClosest point found:")
    print(f"  Face ID: {closest_point['face']}")
    print(f"  Coordinates: ({closest_point['lat']}, {closest_point['lon']})")
    print(f"  Distance: {min_distance:.6f}° (~{min_distance * 111:.2f} km)")
    print(f"  Relative file path: {closest_point['file_path']}")

    # Construct full path to parquet file
    # Path structure: BASE_DATA_DIR/<location>/b4_vap_partition/<relative_path>
    # The file_path in manifest has format: <location>/<partition_path>/<filename>
    # So we need to strip the location prefix and reconstruct with b4_vap_partition
    location = details["location"]
    file_path_str = closest_point["file_path"]

    # Remove location prefix from file path (e.g., "AK_aleutian_islands/lat_deg=..." -> "lat_deg=...")
    if file_path_str.startswith(location + "/"):
        relative_path = file_path_str[len(location) + 1 :]
    else:
        relative_path = file_path_str

    # Construct full path: BASE_DATA_DIR/<location>/b4_vap_partition/<relative_path>
    parquet_file = BASE_DATA_DIR / location / "b4_vap_partition" / relative_path
    print(f"\nReading parquet file: {parquet_file}")

    if not parquet_file.exists():
        raise FileNotFoundError(
            f"Parquet file not found: {parquet_file}\n"
            "Please check BASE_DATA_DIR configuration or data availability."
        )

    # Read parquet file
    df = pd.read_parquet(parquet_file)
    print(f"  Loaded {len(df)} rows × {len(df.columns)} columns")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Time range: {df.index.min()} to {df.index.max()}")

    # Write to output file
    print(f"\nWriting data to: {OUTPUT_FILE}")
    df.to_parquet(OUTPUT_FILE)
    print(f"  Output saved: {OUTPUT_FILE}")

    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)

    return df


def main():
    """
    Main execution function.
    """
    print("=" * 80)
    print("Single Point Tidal Data Extraction")
    print("=" * 80)
    print(f"Query coordinates: ({QUERY_LAT}, {QUERY_LON})")
    print(f"Output file: {OUTPUT_FILE}")
    print()

    # Run analysis
    df = analyze_point(QUERY_LAT, QUERY_LON)

    # Print sample of data
    print("\nData preview (first 5 rows):")
    print(df.head())


if __name__ == "__main__":
    main()
