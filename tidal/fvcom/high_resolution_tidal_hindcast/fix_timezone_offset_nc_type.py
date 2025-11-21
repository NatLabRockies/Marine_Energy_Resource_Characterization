"""
Fix timezone offset units in existing tidal fvcom nc files using h5py.

This script fixes the vap_utc_timezone_offset variable attributes by:
1. Changing units from "hours" to "1" (in-place metadata update)
2. Adding "offset_units" attribute set to "hours" (in-place metadata update)

This is extremely fast as it only modifies attributes, not data.

Usage:
    python fix_timezone_offset_nc_type.py <directory_path>

Example:
    python fix_timezone_offset_nc_type.py /path/to/netcdf/files
"""

import argparse
import sys
from pathlib import Path

import h5py


def fix_timezone_offset_in_file(file_path: Path, dry_run: bool = False) -> dict:
    """
    Fix the vap_utc_timezone_offset variable attributes in a NetCDF4/HDF5 file.

    Parameters
    ----------
    file_path : Path
        Path to the NetCDF4/HDF5 file to fix
    dry_run : bool, optional
        If True, only check what would be changed without modifying files

    Returns
    -------
    dict
        Dictionary with fix status:
        - 'status': 'fixed', 'not_needed', 'no_variable', or 'error'
        - 'message': Description of what happened
        - 'original_dtype': Original dtype if variable exists
        - 'original_units': Original units if variable exists
    """
    result = {
        "status": "unknown",
        "message": "",
        "original_dtype": None,
        "original_units": None,
    }

    try:
        # Open in read mode first to check
        with h5py.File(file_path, "r") as f:
            # Check if variable exists
            if "vap_utc_timezone_offset" not in f:
                result["status"] = "no_variable"
                result["message"] = "Variable vap_utc_timezone_offset not found"
                return result

            var = f["vap_utc_timezone_offset"]

            # Get current dtype and units
            current_dtype = str(var.dtype)
            current_units = var.attrs.get("units", None)
            if isinstance(current_units, bytes):
                current_units = current_units.decode("utf-8")

            result["original_dtype"] = current_dtype
            result["original_units"] = current_units

            # Get current offset_units if it exists
            current_offset_units = var.attrs.get("offset_units", None)
            if isinstance(current_offset_units, bytes):
                current_offset_units = current_offset_units.decode("utf-8")

            # Check what needs fixing
            needs_units_fix = current_units != "1"
            needs_offset_units = current_offset_units != "hours"

            if not needs_units_fix and not needs_offset_units:
                result["status"] = "not_needed"
                result["message"] = (
                    f"Already correct (dtype={current_dtype}, units={current_units}, "
                    f"offset_units={current_offset_units})"
                )
                return result

            if dry_run:
                changes = []
                if needs_units_fix:
                    changes.append(f"units: '{current_units}' -> '1'")
                if needs_offset_units:
                    if current_offset_units is None:
                        changes.append("add offset_units: 'hours'")
                    else:
                        changes.append(
                            f"offset_units: '{current_offset_units}' -> 'hours'"
                        )

                result["status"] = "would_fix"
                result["message"] = f"Would fix: {', '.join(changes)}"
                return result

        # If we get here and not dry_run, modify the file in place
        with h5py.File(file_path, "a") as f:
            var = f["vap_utc_timezone_offset"]

            # Update attributes in place
            if needs_units_fix:
                var.attrs["units"] = "1"

            if needs_offset_units:
                var.attrs["offset_units"] = "hours"

        result["status"] = "fixed"
        changes_made = []
        if needs_units_fix:
            changes_made.append("units='1'")
        if needs_offset_units:
            changes_made.append("offset_units='hours'")
        result["message"] = f"Fixed: {', '.join(changes_made)} (in-place)"

        # Verify the fix
        with h5py.File(file_path, "r") as f_verify:
            var_verify = f_verify["vap_utc_timezone_offset"]
            new_units = var_verify.attrs.get("units", None)
            if isinstance(new_units, bytes):
                new_units = new_units.decode("utf-8")
            new_offset_units = var_verify.attrs.get("offset_units", None)
            if isinstance(new_offset_units, bytes):
                new_offset_units = new_offset_units.decode("utf-8")

            if new_units != "1" or new_offset_units != "hours":
                result["status"] = "error"
                result["message"] = (
                    f"Verification failed: units={new_units}, offset_units={new_offset_units}"
                )

    except Exception as e:
        result["status"] = "error"
        result["message"] = f"Error: {str(e)}"

    return result


def process_directory(directory: Path, dry_run: bool = False) -> None:
    """
    Process all NetCDF files in a directory (non-recursive).

    Parameters
    ----------
    directory : Path
        Directory containing NetCDF4/HDF5 files to process
    dry_run : bool, optional
        If True, only report what would be changed without modifying files
    """
    if not directory.is_dir():
        print(f"Error: {directory} is not a directory")
        sys.exit(1)

    # Find all .nc files in the directory (non-recursive)
    nc_files = sorted(list(directory.glob("*.nc")))

    if not nc_files:
        print(f"No NetCDF files found in {directory}")
        return

    print(f"Found {len(nc_files)} NetCDF files in {directory}")
    if dry_run:
        print("DRY RUN - No files will be modified\n")
    else:
        print()

    # Process each file
    stats = {"fixed": 0, "not_needed": 0, "no_variable": 0, "error": 0, "would_fix": 0}

    for i, nc_file in enumerate(nc_files, 1):
        print(f"[{i}/{len(nc_files)}] Processing: {nc_file.name}")

        result = fix_timezone_offset_in_file(nc_file, dry_run=dry_run)

        print(f"  Status: {result['status']}")
        print(f"  {result['message']}")

        if result["original_dtype"]:
            print(f"  Original dtype: {result['original_dtype']}")
        if result["original_units"]:
            print(f"  Original units: {result['original_units']}")

        print()

        stats[result["status"]] += 1

    # Print summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total files processed: {len(nc_files)}")
    if dry_run:
        print(f"  Would fix: {stats['would_fix']}")
    else:
        print(f"  Fixed: {stats['fixed']}")
    print(f"  Already correct: {stats['not_needed']}")
    print(f"  No variable: {stats['no_variable']}")
    print(f"  Errors: {stats['error']}")
    print()

    if not dry_run and stats["fixed"] > 0:
        print("Note: Files were modified in place (metadata only, extremely fast)")


def main():
    parser = argparse.ArgumentParser(
        description="Fix timezone offset units in NetCDF4/HDF5 files using h5py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run (check what would be changed):
  python fix_timezone_offset_nc_type.py /path/to/files --dry-run

  # Actually fix the files:
  python fix_timezone_offset_nc_type.py /path/to/files

  # Process files in current directory:
  python fix_timezone_offset_nc_type.py .
        """,
    )

    parser.add_argument(
        "directory", type=str, help="Directory containing NetCDF files to process"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check what would be changed without modifying files",
    )

    args = parser.parse_args()

    directory = Path(args.directory).resolve()

    print(f"Processing directory: {directory}")
    print()

    process_directory(directory, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
