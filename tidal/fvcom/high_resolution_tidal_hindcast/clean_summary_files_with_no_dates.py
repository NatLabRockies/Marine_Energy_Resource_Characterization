"""One-off script to find and delete b4/b5 output files with broken date fields.

The bug: commit 1b55214 added version numbers to b3 NC filenames, which caused
the downstream b4/b5 filename parser (split(".")[-3:-1]) to pick up "0.0" from
the version string instead of the actual date. This script finds those bad files
and deletes them with confirmation.

Bad pattern:  *.b4.0.0.v1.0.0.*  or  *.b5.0.0.v1.0.0.*
Good pattern: *.b4.20100603.000000.v1.0.0.*  or  *.b5.20100603.000000.v1.0.0.*

Can be run from anywhere.
"""

import sys
from pathlib import Path

from config import config

# HPC base path where all output data lives
BASE_DIR = Path(config["dir"]["base"])
VERSION = f"v{config['dataset']['version']}"
LOCATIONS = config["location_specification"]


def find_bad_files():
    """Find all files with broken date fields (date=0, time=0) in b4/b5 directories."""
    bad_files = []

    for location in LOCATIONS.values():
        loc_name = location["output_name"]

        # Check b4 and b5 directories (including gis/ subdirectories)
        search_dirs = [
            BASE_DIR / loc_name / VERSION / "b4_vap_summary_parquet",
            BASE_DIR / loc_name / VERSION / "b5_vap_atlas_summary_parquet",
        ]

        for search_dir in search_dirs:
            if not search_dir.exists():
                continue

            # Recursively find files with the bad date pattern
            for f in search_dir.rglob("*"):
                if not f.is_file():
                    continue
                # Bad files have .b4.0.0. or .b5.0.0. in the name
                if ".b4.0.0." in f.name or ".b5.0.0." in f.name:
                    bad_files.append(f)

    return sorted(bad_files)


def main():
    print(f"Searching: {BASE_DIR}")
    print(f"Version:   {VERSION}\n")

    bad_files = find_bad_files()

    if not bad_files:
        print("No files with broken date fields found.")
        return

    print(f"Found {len(bad_files)} file(s) with broken date fields (date=0, time=0):\n")
    for i, f in enumerate(bad_files, 1):
        print(f"  {i}. {f}")

    print()

    deleted = 0
    skipped = 0

    for f in bad_files:
        response = input(f"Delete {f.name}? [y/N] ").strip().lower()
        if response == "y":
            f.unlink()
            print(f"  Deleted: {f}")
            deleted += 1
        else:
            print(f"  Skipped: {f}")
            skipped += 1

    print(f"\nDone. Deleted: {deleted}, Skipped: {skipped}")


if __name__ == "__main__":
    main()
