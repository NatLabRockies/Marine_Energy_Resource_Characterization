import os
import re
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import xarray as xr

from src.attrs_manager import standardize_dataset_global_attrs


# Global configuration
START_DIRECTORY = "/home/asimms/.conda-envs/tidal_fvcom/high_resolution_tidal_hindcast"
EXCLUDE_DIRECTORIES = ["00_raw"]  # Configurable list of directories to skip
ATTRS_BACKUP_DIR = "attrs_backups"  # Directory to store attribute backups


def parse_me_filename(filename: str) -> Dict[str, Optional[str]]:
    """
    Parse ME naming convention filename and extract components.

    Expected format: [prefix.]location_id.dataset_name[-qualifier][-temporal].data_level.date.time.ext

    Args:
        filename: Filename to parse

    Returns:
        Dictionary with extracted components
    """
    # Remove extension
    name_parts = filename.split(".")

    if len(name_parts) < 6:  # Minimum: location.dataset.level.date.time.ext
        return {"error": f"Filename {filename} doesn't match expected format"}

    # Handle optional prefix (like "001" in your example)
    start_idx = 0
    if name_parts[0].isdigit():
        prefix = name_parts[0]
        start_idx = 1
    else:
        prefix = None

    # Extract core components
    location_id = name_parts[start_idx]
    dataset_name_full = name_parts[start_idx + 1]
    data_level = name_parts[start_idx + 2]
    date_str = name_parts[start_idx + 3]
    time_str = name_parts[start_idx + 4]
    ext = name_parts[-1]

    # Parse dataset_name for qualifier and temporal
    dataset_name = dataset_name_full
    qualifier = None
    temporal = None

    # Look for temporal suffix (ends with time unit like 'h', 's', 'm')
    temporal_pattern = r"-(\d+[hsmHSM][z]?)$"
    temporal_match = re.search(temporal_pattern, dataset_name_full)
    if temporal_match:
        temporal = temporal_match.group(1)
        dataset_name = dataset_name_full[: temporal_match.start()]

    # Look for qualifier (anything else after the first dash)
    if "-" in dataset_name and not temporal:
        parts = dataset_name.split("-", 1)
        dataset_name = parts[0]
        qualifier = parts[1]
    elif "-" in dataset_name and temporal:
        # Remove temporal part and check for qualifier
        base_name = dataset_name_full[: temporal_match.start()]
        if "-" in base_name:
            parts = base_name.split("-", 1)
            dataset_name = parts[0]
            qualifier = parts[1]

    return {
        "prefix": prefix,
        "location_id": location_id,
        "dataset_name": dataset_name,
        "qualifier": qualifier,
        "temporal": temporal,
        "data_level": data_level,
        "date": date_str,
        "time": time_str,
        "extension": ext,
        "full_filename": filename,
    }


def create_attrs_backup_structure():
    """Create directory structure for attribute backups."""
    backup_path = Path(ATTRS_BACKUP_DIR)
    backup_path.mkdir(exist_ok=True)
    return backup_path


def get_attrs_backup_filepath(location_id: str, data_level: str, filename: str) -> Path:
    """
    Generate backup filepath for attributes organized by location and data level.

    Args:
        location_id: Location identifier from filename
        data_level: Data level from filename
        filename: Original filename

    Returns:
        Path to JSON backup file
    """
    backup_root = create_attrs_backup_structure()
    location_dir = backup_root / location_id
    data_level_dir = location_dir / data_level

    # Create directories if they don't exist
    data_level_dir.mkdir(parents=True, exist_ok=True)

    # Use original filename but with .json extension
    json_filename = Path(filename).stem + "_attrs.json"
    return data_level_dir / json_filename


def backup_attrs_to_json(
    attrs: Dict, location_id: str, data_level: str, filename: str
) -> Path:
    """
    Backup attributes to JSON file organized by location/data_level.

    Args:
        attrs: Dictionary of attributes to backup
        location_id: Location identifier
        data_level: Data level
        filename: Original filename

    Returns:
        Path to created backup file
    """
    backup_filepath = get_attrs_backup_filepath(location_id, data_level, filename)

    # Convert attributes to JSON-serializable format
    json_attrs = {}
    for key, value in attrs.items():
        if hasattr(value, "tolist"):  # numpy arrays
            json_attrs[key] = value.tolist()
        elif hasattr(value, "item"):  # numpy scalars
            json_attrs[key] = value.item()
        else:
            json_attrs[key] = value

    # Add metadata about the backup
    backup_data = {
        "backup_timestamp": datetime.now().isoformat(),
        "original_filename": filename,
        "location_id": location_id,
        "data_level": data_level,
        "attributes": json_attrs,
    }

    # Write to JSON file
    with open(backup_filepath, "w") as f:
        json.dump(backup_data, f, indent=2)

    return backup_filepath


def restore_attrs_from_json(
    location_id: str, data_level: str, filename: str
) -> Optional[Dict]:
    """
    Restore attributes from JSON backup file.

    Args:
        location_id: Location identifier
        data_level: Data level
        filename: Original filename

    Returns:
        Dictionary of restored attributes or None if backup doesn't exist
    """
    backup_filepath = get_attrs_backup_filepath(location_id, data_level, filename)

    if not backup_filepath.exists():
        return None

    try:
        with open(backup_filepath, "r") as f:
            backup_data = json.load(f)

        return backup_data.get("attributes", {})

    except Exception as e:
        print(f"Error reading backup file {backup_filepath}: {e}")
        return None


def should_skip_directory(dir_path: Path) -> bool:
    """Check if directory should be skipped based on exclude list."""
    return dir_path.name in EXCLUDE_DIRECTORIES


def find_nc_files(start_dir: str) -> List[Tuple[str, Dict]]:
    """
    Find all .nc files in directory tree and parse their filenames.

    Args:
        start_dir: Root directory to search

    Returns:
        List of tuples: (file_path, parsed_filename_dict)
    """
    nc_files = []
    start_path = Path(start_dir)

    for root, dirs, files in os.walk(start_path):
        # Skip excluded directories
        root_path = Path(root)
        if should_skip_directory(root_path):
            continue

        # Remove excluded directories from dirs to prevent walking into them
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRECTORIES]

        # Process .nc files
        for file in files:
            if file.endswith(".nc"):
                file_path = os.path.join(root, file)
                parsed = parse_me_filename(file)
                nc_files.append((file_path, parsed))

    return nc_files


def update_attrs_inplace_h5py(file_path: str, new_attrs: Dict) -> None:
    """
    Update NetCDF4 file attributes in-place using h5py.

    Args:
        file_path: Path to the NetCDF file
        new_attrs: Dictionary of new attributes to set
    """
    import h5py

    # Open NetCDF4 file with h5py in read-write mode
    with h5py.File(file_path, "r+") as f:
        # Clear existing global attributes
        existing_attrs = list(f.attrs.keys())
        for attr_name in existing_attrs:
            del f.attrs[attr_name]

        # Set new attributes directly
        for key, value in new_attrs.items():
            # Handle different data types appropriately
            if isinstance(value, str):
                # Ensure string attributes are properly encoded
                f.attrs[key] = value
            elif isinstance(value, (int, float, bool)):
                f.attrs[key] = value
            elif hasattr(value, "__iter__") and not isinstance(value, str):
                # Handle arrays/lists
                f.attrs[key] = value
            else:
                # Convert other types to string
                f.attrs[key] = str(value)


def process_single_file(file_path: str, parsed_info: Dict, config=None) -> bool:
    """
    Process a single NetCDF file with standardize_dataset_global_attrs.

    Args:
        file_path: Path to the NetCDF file
        parsed_info: Parsed filename information
        config: Configuration object (placeholder)

    Returns:
        True if successful, False otherwise
    """
    try:
        print(f"Processing: {file_path}")
        print(f"  Location: {parsed_info['location_id']}")
        print(f"  Data Level: {parsed_info['data_level']}")
        print(f"  Date: {parsed_info['date']}")

        # Read the dataset
        ds = xr.open_dataset(file_path)

        # Store original attributes for backup
        original_attrs = dict(ds.attrs)
        filename = Path(file_path).name

        # Create backup of original attributes
        backup_path = backup_attrs_to_json(
            original_attrs,
            parsed_info["location_id"],
            parsed_info["data_level"],
            filename,
        )
        print(f"  Backed up attributes to: {backup_path}")

        # Apply the standardization function
        ds = standardize_dataset_global_attrs(
            ds=ds,
            config=config,
            location=parsed_info["location_id"],
            data_level=parsed_info["data_level"],
            input_files=[],  # Empty list as requested
            input_ds_is_original_model_output=False,
            # coordinate_reference_system_string=None,
        )

        # Check if attributes were modified
        attrs_modified = ds.attrs != original_attrs

        if attrs_modified:
            print("  Attributes modified, updating file...")

            try:
                # Update attributes in-place using h5py (most efficient)
                update_attrs_inplace_h5py(file_path, ds.attrs)
                print("  Successfully updated attributes in-place using h5py")

            except Exception as e:
                print(f"  ERROR: In-place update failed: {e}")
                print(f"  Attributes can be restored from: {backup_path}")

                # Close dataset before returning
                ds.close()
                return False

        else:
            print("  No attribute changes needed")

        # Close the dataset
        ds.close()
        return True

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False


def restore_file_attrs(
    location_id: str, data_level: str, filename: str, file_path: str
) -> bool:
    """
    Utility function to restore attributes from backup.

    Args:
        location_id: Location identifier
        data_level: Data level
        filename: Original filename
        file_path: Path to NetCDF file

    Returns:
        True if restoration successful
    """
    print(f"Restoring attributes for: {file_path}")

    # Get backed up attributes
    original_attrs = restore_attrs_from_json(location_id, data_level, filename)

    if original_attrs is None:
        print(f"  No backup found for {filename}")
        return False

    try:
        # Restore attributes using h5py
        update_attrs_inplace_h5py(file_path, original_attrs)
        print("  Successfully restored attributes")
        return True

    except Exception as e:
        print(f"  Error restoring attributes: {e}")
        return False


def confirm_operation() -> bool:
    """
    Ask user to confirm the operation before proceeding.

    Returns:
        True if user confirms, False otherwise
    """
    print("\n" + "=" * 60)
    print("NETCDF ATTRIBUTE PROCESSING CONFIRMATION")
    print("=" * 60)
    print(f"Start directory: {START_DIRECTORY}")
    print(f"Excluded directories: {EXCLUDE_DIRECTORIES}")
    print(f"Backup directory: {ATTRS_BACKUP_DIR}")
    print("\nThis operation will:")
    print("  1. Find all .nc files in the directory tree")
    print("  2. Backup original attributes to JSON files")
    print("  3. Modify global attributes in-place using h5py")
    print("  4. Create organized backups by location/data_level")
    print("\nWARNING: This will modify NetCDF files in-place!")
    print("Make sure you have adequate backups of your data.")
    print("=" * 60)

    while True:
        response = input("\nDo you want to proceed? (yes/no): ").strip().lower()
        if response in ["yes", "y"]:
            return True
        elif response in ["no", "n"]:
            return False
        else:
            print("Please enter 'yes' or 'no'")


def main():
    """Main processing function."""
    print("NetCDF Global Attributes Processor")
    print(f"Starting directory: {START_DIRECTORY}")
    print(f"Excluding directories: {EXCLUDE_DIRECTORIES}")
    print(f"Attribute backups will be stored in: {ATTRS_BACKUP_DIR}")

    # Check if start directory exists
    if not os.path.exists(START_DIRECTORY):
        print(f"\nERROR: Start directory does not exist: {START_DIRECTORY}")
        return

    # Find all NC files first to show user what will be processed
    print("\nScanning for NetCDF files...")
    nc_files = find_nc_files(START_DIRECTORY)
    print(f"Found {len(nc_files)} NetCDF files")

    if len(nc_files) == 0:
        print("No NetCDF files found. Exiting.")
        return

    # Show sample of files that will be processed
    print("\nSample files to be processed:")
    for i, (file_path, parsed_info) in enumerate(nc_files[:5]):
        if "error" not in parsed_info:
            print(
                f"  {i+1}. {Path(file_path).name} (Location: {parsed_info['location_id']}, Level: {parsed_info['data_level']})"
            )
        else:
            print(f"  {i+1}. {Path(file_path).name} (PARSE ERROR)")

    if len(nc_files) > 5:
        print(f"  ... and {len(nc_files) - 5} more files")

    # Ask for confirmation
    if not confirm_operation():
        print("\nOperation cancelled by user.")
        return

    print("\nProceeding with processing...")

    # Create backup directory structure
    create_attrs_backup_structure()

    # Process each file
    successful = 0
    failed = 0

    for file_path, parsed_info in nc_files:
        if "error" in parsed_info:
            print(f"Skipping {file_path}: {parsed_info['error']}")
            failed += 1
            continue

        if process_single_file(file_path, parsed_info):
            successful += 1
        else:
            failed += 1

    print("\nProcessing complete:")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Attribute backups stored in: {Path(ATTRS_BACKUP_DIR).absolute()}")


if __name__ == "__main__":
    main()
