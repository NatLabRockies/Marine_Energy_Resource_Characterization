"""
Check if a query point is within 100 meters of actual data using the tidal manifest.

This script uses the TidalManifestQuery class to find the nearest data point
and reports whether it's within 100 meters.

Usage:
    cd tidal/fvcom/high_resolution_tidal_hindcast && python check_point_distance.py
"""

from pathlib import Path
from query_tidal_manifest import TidalManifestQuery


def check_point_distance(lat, lon, max_distance_meters=100):
    """
    Check if query point is within specified distance of actual data.

    Parameters
    ----------
    lat : float
        Query latitude
    lon : float
        Query longitude
    max_distance_meters : float
        Maximum acceptable distance in meters (default: 100)

    Returns
    -------
    dict
        Result with keys:
        - within_threshold: bool
        - distance_meters: float
        - distance_km: float
        - nearest_point: dict (if found)
    """
    # Manifest path on HPC
    manifest_path = Path(
        "/projects/hindcastra/Tidal/datasets/high_resolution_tidal_hindcast/manifests/v0.3.0/manifest.json"
    )

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found at {manifest_path}")

    # Initialize query interface
    print(f"Loading manifest from: {manifest_path}")
    query = TidalManifestQuery(manifest_path)

    print(f"\nQuerying nearest point to ({lat}, {lon})...")

    # Query nearest point
    result = query.query_nearest_point(lat=lat, lon=lon, load_details=False)

    if result is None:
        print("No nearby points found!")
        return {
            "within_threshold": False,
            "distance_meters": None,
            "distance_km": None,
            "nearest_point": None,
        }

    # Convert distance to meters
    distance_km = result["distance_km"]
    distance_meters = distance_km * 1000.0

    # Check threshold
    within_threshold = distance_meters <= max_distance_meters

    return {
        "within_threshold": within_threshold,
        "distance_meters": distance_meters,
        "distance_km": distance_km,
        "nearest_point": result,
    }


def main():
    """Main function to check specific point."""
    # Query point
    query_lat = 43.05067443847656
    query_lon = -70.7033920288086
    threshold_meters = 100

    print("=" * 80)
    print("Point Distance Check")
    print("=" * 80)
    print(f"\nQuery Point:")
    print(f"  Latitude:  {query_lat}")
    print(f"  Longitude: {query_lon}")
    print(f"\nThreshold: {threshold_meters} meters")
    print("=" * 80)

    # Check distance
    result = check_point_distance(query_lat, query_lon, max_distance_meters=threshold_meters)

    # Print results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    if result["nearest_point"] is None:
        print("\n❌ No data points found near query location")
        return

    nearest = result["nearest_point"]["point"]
    distance_m = result["distance_meters"]
    distance_km = result["distance_km"]

    print(f"\nNearest Data Point:")
    print(f"  Latitude:  {nearest['lat']}")
    print(f"  Longitude: {nearest['lon']}")
    print(f"  Face ID:   {nearest['face_id']}")
    print(f"  Location:  {result['nearest_point']['location']}")

    print(f"\nDistance from Query Point:")
    print(f"  {distance_m:.2f} meters")
    print(f"  {distance_km:.6f} km")

    print(f"\nThreshold Check:")
    if result["within_threshold"]:
        print(f"  ✅ PASS - Point is within {threshold_meters} meters")
        print(f"     ({distance_m:.2f} m < {threshold_meters} m)")
    else:
        print(f"  ❌ FAIL - Point exceeds {threshold_meters} meters")
        print(f"     ({distance_m:.2f} m > {threshold_meters} m)")

    print(f"\nFile Path:")
    print(f"  {nearest['file_path']}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()