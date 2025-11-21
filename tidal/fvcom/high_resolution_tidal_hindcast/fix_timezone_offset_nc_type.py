"""
Fix timezone offset dtype and units in existing NetCDF files.

This script fixes the vap_utc_timezone_offset variable in NetCDF files by:
1. Converting dtype from timedelta64[ns] to int16
2. Changing units from "hours" to "1" (dimensionless) to prevent xarray's
   automatic timedelta decoding
3. Adding "offset_units" attribute set to "hours" for human interpretation

Usage:
    python fix_timezone_offset_nc_type.py <directory_path>

Example:
    python fix_timezone_offset_nc_type.py /path/to/netcdf/files
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import xarray as xr


def fix_timezone_offset_in_file(file_path: Path, dry_run: bool = False) -> dict:
    """
    Fix the vap_utc_timezone_offset variable in a NetCDF file.

    Parameters
    ----------
    file_path : Path
        Path to the NetCDF file to fix
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
        # Open dataset in read mode first to check
        with xr.open_dataset(file_path, decode_timedelta=False) as ds:
            # Check if variable exists
            if "vap_utc_timezone_offset" not in ds.variables:
                result["status"] = "no_variable"
                result["message"] = "Variable vap_utc_timezone_offset not found"
                return result

            # Get current dtype and units
            current_dtype = str(ds.vap_utc_timezone_offset.dtype)
            current_units = ds.vap_utc_timezone_offset.attrs.get("units", None)

            result["original_dtype"] = current_dtype
            result["original_units"] = current_units

            # Check if already fixed
            needs_dtype_fix = current_dtype != "int16"
            needs_units_fix = current_units != "1"
            needs_offset_units = "offset_units" not in ds.vap_utc_timezone_offset.attrs

            if not needs_dtype_fix and not needs_units_fix and not needs_offset_units:
                result["status"] = "not_needed"
                result["message"] = (
                    f"Already correct (dtype={current_dtype}, units={current_units})"
                )
                return result

            if dry_run:
                changes = []
                if needs_dtype_fix:
                    changes.append(f"dtype: {current_dtype} -> int16")
                if needs_units_fix:
                    changes.append(f"units: {current_units} -> 1")
                if needs_offset_units:
                    changes.append("add offset_units: hours")

                result["status"] = "would_fix"
                result["message"] = f"Would fix: {', '.join(changes)}"
                return result

        # If we get here and not dry_run, we need to modify the file
        # Load the full dataset
        ds = xr.open_dataset(file_path, decode_timedelta=False)

        # Get the variable values as int16
        tz_values = ds.vap_utc_timezone_offset.values

        # If dtype is timedelta64, convert to hours (int64) first, then to int16
        if np.issubdtype(tz_values.dtype, np.timedelta64):
            # Convert timedelta64[ns] to hours (as integers)
            tz_values = (tz_values / np.timedelta64(1, "h")).astype(np.int16)
        else:
            # Just ensure it's int16
            tz_values = tz_values.astype(np.int16)

        # Store original attributes
        original_attrs = ds.vap_utc_timezone_offset.attrs.copy()

        # Create new variable with correct dtype
        ds["vap_utc_timezone_offset"] = xr.DataArray(
            tz_values,
            dims=ds.vap_utc_timezone_offset.dims,
            coords=ds.vap_utc_timezone_offset.coords,
        )

        # Update attributes
        original_attrs["units"] = "1"  # Dimensionless
        original_attrs["offset_units"] = "hours"  # For human interpretation

        ds["vap_utc_timezone_offset"].attrs = original_attrs

        # Write to temporary file first
        temp_path = file_path.with_suffix(".nc.tmp")

        # Write the fixed dataset to temp file
        ds.to_netcdf(temp_path, engine="netcdf4")

        # Close the dataset
        ds.close()

        # Verify temp file before replacing original
        try:
            with xr.open_dataset(temp_path, decode_timedelta=False) as ds_verify_temp:
                verify_dtype = str(ds_verify_temp.vap_utc_timezone_offset.dtype)
                verify_units = ds_verify_temp.vap_utc_timezone_offset.attrs.get("units")

                if verify_dtype != "int16" or verify_units != "1":
                    temp_path.unlink()  # Delete bad temp file
                    result["status"] = "error"
                    result["message"] = (
                        f"Verification failed before replacing original: "
                        f"dtype={verify_dtype}, units={verify_units}"
                    )
                    return result
        except Exception as e:
            if temp_path.exists():
                temp_path.unlink()  # Clean up temp file
            result["status"] = "error"
            result["message"] = f"Verification error: {str(e)}"
            return result

        # Replace original file with fixed version (no backup)
        temp_path.replace(file_path)

        result["status"] = "fixed"
        result["message"] = (
            f"Fixed: dtype {result['original_dtype']} -> int16, "
            f"units '{result['original_units']}' -> '1'"
        )

        # Verify the fix
        with xr.open_dataset(file_path, decode_timedelta=False) as ds_verify:
            new_dtype = str(ds_verify.vap_utc_timezone_offset.dtype)
            new_units = ds_verify.vap_utc_timezone_offset.attrs.get("units")

            if new_dtype != "int16" or new_units != "1":
                result["status"] = "error"
                result["message"] = (
                    f"Verification failed after fix: dtype={new_dtype}, units={new_units}"
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
        Directory containing NetCDF files to process
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
        print("Note: Files were modified in place (no backups created)")


def main():
    parser = argparse.ArgumentParser(
        description="Fix timezone offset dtype and units in NetCDF files",
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
