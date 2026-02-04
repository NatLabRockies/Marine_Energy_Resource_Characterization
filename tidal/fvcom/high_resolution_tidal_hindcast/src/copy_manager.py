import os
import shutil
import time
from pathlib import Path
import tarfile
import zipfile
import datetime
import psutil
import threading
import subprocess


def scan_directory(input_dir, show_progress=True):
    """
    Scan a directory to count files and estimate total size using CLI commands.
    Much faster for massive directories than walking the directory tree in Python.

    Parameters:
    -----------
    input_dir : str or Path
        Directory to scan
    show_progress : bool
        Whether to show progress during scanning

    Returns:
    --------
    dict
        Dictionary containing scan statistics
    """
    input_dir = Path(input_dir).resolve()
    print(f"Scanning source directory: {input_dir}")
    scan_start = time.time()

    # Use find command to count files (much faster than Python for massive directories)
    print("Counting files using find command...")
    try:
        # Count files using find command
        # Use shell=True to allow pipe redirection
        count_result = subprocess.run(
            f"find {str(input_dir)} -type f -printf '.' | wc -c",
            shell=True,
            text=True,
            capture_output=True,
        )

        if count_result.returncode == 0:
            file_count = int(count_result.stdout.strip())
            print(f"File count from find command: {file_count:,}")
        else:
            print(f"Error using find command: {count_result.stderr}")
            file_count = 0

        # Estimate total size by sampling files
        # This is much faster than trying to get the exact size of every file
        print("Estimating total size by sampling...")

        # Sample a subset of files to estimate average size
        # For very large directories, we'll sample up to 1000 files
        sample_cmd = f"find {str(input_dir)} -type f -print | head -n 1000"
        sample_result = subprocess.run(
            sample_cmd, shell=True, text=True, capture_output=True
        )

        if sample_result.returncode == 0:
            sample_files = sample_result.stdout.strip().split("\n")
            sample_count = len(sample_files)

            if sample_count > 0:
                # Calculate total size of sampled files
                sample_size = 0
                for file_path in sample_files:
                    if file_path.strip():  # Skip empty lines
                        try:
                            sample_size += os.path.getsize(file_path)
                        except (FileNotFoundError, PermissionError):
                            pass

                # Calculate average file size and estimate total
                if sample_count > 0:
                    avg_file_size = sample_size / sample_count
                    total_size = file_count * avg_file_size
                    print(
                        f"Average file size (from {sample_count:,} samples): {format_size(avg_file_size)}"
                    )
                else:
                    avg_file_size = 0
                    total_size = 0
            else:
                avg_file_size = 0
                total_size = 0
        else:
            print(f"Error sampling files: {sample_result.stderr}")
            avg_file_size = 0
            total_size = 0

        # Try to get a more accurate size using du if possible
        # This might be slow for very large directories
        print("Attempting to get more accurate size using du command...")
        du_cmd = f"du -sb {str(input_dir)}"
        du_result = subprocess.run(
            du_cmd, shell=True, text=True, capture_output=True, timeout=10
        )

        if du_result.returncode == 0:
            try:
                # du output format: "size path"
                du_size = int(du_result.stdout.split()[0])
                print(f"Total size from du command: {format_size(du_size)}")
                total_size = du_size
            except (IndexError, ValueError):
                print(f"Error parsing du output: {du_result.stdout}")
        else:
            print("du command failed or timed out, using estimate from sampling")

    except subprocess.TimeoutExpired:
        print(
            "Command timed out - directory is too large for accurate size measurement"
        )
    except Exception as e:
        print(f"Error scanning directory: {e}")
        file_count = 0
        total_size = 0
        avg_file_size = 0

    scan_time = time.time() - scan_start

    # Create result dictionary
    scan_results = {
        "file_count": file_count,
        "total_size": total_size,
        "scan_time": scan_time,
        "avg_file_size": avg_file_size if file_count > 0 else 0,
        "is_estimated": True,  # Size is estimated from sampling
    }

    # Print summary
    print(f"Scan complete in {format_time(scan_time)}")
    print(f"Files found: {file_count:,}")
    print(f"Total size: {format_size(total_size)}")

    return scan_results


def copy_directory(
    input_dir,
    output_dir,
    method="rsync",
    compression_level=1,
    show_progress=True,
):
    """
    Copy a directory from input to output location, with options for different transfer methods.

    Parameters:
    -----------
    input_dir : str or Path
        Source directory to copy from
    output_dir : str or Path
        Destination directory to copy to
    method : str
        Method to use for copying: "tar", "zip", or "rsync"
    compression_level : int
        Compression level for tar/zip (1-9, higher = more compression but slower)
    chunk_size : int
        Size of chunks to use when copying files (in KB)
    show_progress : bool
        Whether to show progress during the operation

    Returns:
    --------
    dict
        Dictionary containing statistics about the operation
    """
    # Convert to Path objects
    input_dir = Path(input_dir).resolve()
    output_dir = Path(output_dir).resolve()

    # Validate inputs
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Start timer
    start_time = time.time()

    # Print operation details
    print(f"\n{'='*80}")
    print("FILE TRANSFER OPERATION")
    print(f"{'='*80}")
    print(f"Started at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Source: {input_dir}")
    print(f"Destination: {output_dir}")
    print(f"Method: {method}")

    # Scan directory for stats using CLI commands (much faster)
    scan_results = scan_directory(input_dir, show_progress)

    # Statistics dictionary
    stats = {
        "method": method,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "start_time": start_time,
        "file_count": scan_results["file_count"],
        "total_size": scan_results["total_size"],
        "files_processed": 0,
        "bytes_processed": 0,
    }

    # Execute the selected transfer method
    if method == "tar":
        _tar_copy(input_dir, output_dir, stats, compression_level, show_progress)
    elif method == "zip":
        _zip_copy(input_dir, output_dir, stats, compression_level, show_progress)
    elif method == "rsync":
        _rsync_copy(input_dir, output_dir, stats, show_progress)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Finish and return stats
    end_time = time.time()
    elapsed = end_time - start_time
    transfer_rate = stats["bytes_processed"] / elapsed if elapsed > 0 else 0

    # Print summary
    print(f"\n{'='*80}")
    print("TRANSFER COMPLETE")
    print(f"{'='*80}")
    print(f"Completed at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Elapsed time: {format_time(elapsed)}")
    print(f"Files processed: {stats['files_processed']:,} / {stats['file_count']:,}")
    print(
        f"Data transferred: {format_size(stats['bytes_processed'])} / {format_size(stats['total_size'])}"
    )
    print(f"Transfer rate: {format_size(transfer_rate)}/s")

    # Verify file count in destination if not using archive methods
    if method in ["direct", "rsync"]:
        print("\nVerifying file counts...")
        try:
            # Use the find command to count files in destination
            verify_cmd = f"find {str(output_dir)} -type f -printf '.' | wc -c"
            verify_result = subprocess.run(
                verify_cmd, shell=True, text=True, capture_output=True
            )

            if verify_result.returncode == 0:
                dest_file_count = int(verify_result.stdout.strip())
                print(f"Source files: {stats['file_count']:,}")
                print(f"Destination files: {dest_file_count:,}")

                if dest_file_count == stats["file_count"]:
                    print("✓ File counts match")
                else:
                    print(
                        f"⚠ WARNING: File counts don't match! Difference: {abs(dest_file_count - stats['file_count']):,}"
                    )
            else:
                print(f"Error verifying file count: {verify_result.stderr}")

        except Exception as e:
            print(f"Error verifying file counts: {e}")

    # Calculate and display system resource usage
    cpu_percent = psutil.cpu_percent()
    memory_info = psutil.virtual_memory()
    print("\nSystem resource usage:")
    print(f"CPU utilization: {cpu_percent}%")
    print(
        f"Memory usage: {format_size(memory_info.used)} / {format_size(memory_info.total)} ({memory_info.percent}%)"
    )

    return stats


def _tar_copy(input_dir, output_dir, stats, compression_level, show_progress):
    """Create a tar archive, transfer it, and extract at destination"""
    print(
        "\nStarting tar archive transfer (not recommended for massive directories)..."
    )

    # For massive directories, warn that this might not be the best option
    print(
        "WARNING: TAR method may be slow or fail for directories with millions of files."
    )
    print("Consider using 'direct' or 'rsync' method instead.")

    archive_name = f"{input_dir.name}.tar.gz"
    archive_path = Path(output_dir.parent) / archive_name

    print(f"Creating streaming archive to: {archive_path}")

    # Use subprocess to call tar directly for better performance
    tar_cmd = [
        "tar",
        f"-{compression_level}cf",  # Compression level + create + file
        str(archive_path),
        "-C",
        str(input_dir.parent),  # Change to parent directory
        input_dir.name,  # Add only the target directory
    ]

    # Run tar command
    print(f"Running: {' '.join(tar_cmd)}")
    tar_process = subprocess.run(tar_cmd, capture_output=True, text=True)

    if tar_process.returncode != 0:
        print(f"Error creating archive: {tar_process.stderr}")
        raise RuntimeError(f"Failed to create archive: {tar_process.stderr}")

    # Update stats to indicate all files processed for archiving
    stats["files_processed"] = stats["file_count"]
    stats["bytes_processed"] = stats["total_size"]

    # Extract at destination
    print(f"Extracting archive to: {output_dir.parent}")

    # Build extract command
    extract_cmd = [
        "tar",
        "-xf",  # Extract + file
        str(archive_path),
        "-C",
        str(output_dir.parent),  # Change to parent directory
    ]

    # Run extract command
    print(f"Running: {' '.join(extract_cmd)}")
    extract_process = subprocess.run(extract_cmd, capture_output=True, text=True)

    if extract_process.returncode != 0:
        print(f"Error extracting archive: {extract_process.stderr}")
        raise RuntimeError(f"Failed to extract archive: {extract_process.stderr}")

    # Remove the archive
    print("Cleaning up temporary archive")
    archive_path.unlink()


def _zip_copy(input_dir, output_dir, stats, compression_level, show_progress):
    """Create a zip archive, transfer it, and extract at destination"""
    print(
        "\nStarting zip archive transfer (not recommended for massive directories)..."
    )

    # For massive directories, warn that this might not be the best option
    print(
        "WARNING: ZIP method may be very slow or fail for directories with millions of files."
    )
    print("Consider using 'direct' or 'rsync' method instead.")

    archive_name = f"{input_dir.name}.zip"
    archive_path = Path(output_dir.parent) / archive_name

    # Use external zip command if available for better performance
    if shutil.which("zip") and shutil.which("unzip"):
        # Use external zip command
        print("Using system zip command for better performance")

        # Build zip command
        zip_cmd = [
            "zip",
            f"-{compression_level}r",  # Compression level + recursive
            str(archive_path),  # Output file
            str(input_dir),  # Input directory
        ]

        # Run zip command
        print(f"Running: {' '.join(zip_cmd)}")
        zip_process = subprocess.run(zip_cmd, capture_output=True, text=True)

        if zip_process.returncode != 0:
            print(f"Error creating zip archive: {zip_process.stderr}")
            raise RuntimeError(f"Failed to create zip archive: {zip_process.stderr}")

        # Update stats
        stats["files_processed"] = stats["file_count"]
        stats["bytes_processed"] = stats["total_size"]

        # Extract at destination
        print(f"Extracting archive to: {output_dir.parent}")

        # Build unzip command
        unzip_cmd = [
            "unzip",
            "-o",  # Overwrite existing files without prompting
            str(archive_path),  # Input archive
            "-d",
            str(output_dir.parent),  # Output directory
        ]

        # Run unzip command
        print(f"Running: {' '.join(unzip_cmd)}")
        unzip_process = subprocess.run(unzip_cmd, capture_output=True, text=True)

        if unzip_process.returncode != 0:
            print(f"Error extracting zip archive: {unzip_process.stderr}")
            raise RuntimeError(f"Failed to extract zip archive: {unzip_process.stderr}")
    else:
        print(
            "Error: zip/unzip commands not found. ZIP method requires system zip/unzip commands."
        )
        print("Please install zip/unzip or use a different transfer method.")
        raise RuntimeError("zip/unzip commands not available")

    # Remove the archive
    print("Cleaning up temporary archive")
    archive_path.unlink()


def _rsync_copy(input_dir, output_dir, stats, show_progress):
    """Use rsync for optimized copying (Unix/Linux systems only)"""
    if not shutil.which("rsync"):
        print(
            "Error: rsync command not found. RSYNC method requires rsync to be installed."
        )
        print("Please install rsync or use a different transfer method.")
        raise RuntimeError("rsync command not available")

    print("\nStarting rsync transfer (best option for massive directories)...")

    # Build rsync command with optimizations for massive transfers
    rsync_cmd = [
        "rsync",
        "-a",  # Archive mode (preserves permissions, timestamps, etc.)
        "-H",  # Preserve hard links
        "--numeric-ids",  # Don't map user/group IDs
        "--no-i-r",  # Don't use incremental recursion (better for large dir trees)
        "--info=progress2" if show_progress else "",  # Show progress
        "--stats",  # Show statistics at the end
        "--exclude='.DS_Store'",  # Exclude macOS metadata files
        "--exclude='*.swp'",  # Exclude vim swap files
        "--exclude='*~'",  # Exclude backup files
        str(input_dir) + "/",  # Source with trailing slash (contents only)
        str(output_dir),  # Destination
    ]

    # Filter empty arguments
    rsync_cmd = [arg for arg in rsync_cmd if arg]

    # Run rsync as shell command to properly handle wildcards in exclude patterns
    rsync_shell_cmd = " ".join(rsync_cmd)

    # Run rsync
    print(f"Running: {rsync_shell_cmd}")
    process = subprocess.run(
        rsync_shell_cmd, shell=True, text=True, capture_output=True
    )

    # Print rsync output
    if process.stdout:
        print(process.stdout)

    if process.stderr:
        print("Errors:")
        print(process.stderr)

    # Update stats based on rsync output (simplified)
    if process.returncode == 0:
        stats["files_processed"] = stats["file_count"]
        stats["bytes_processed"] = stats["total_size"]
    else:
        print(f"rsync failed with return code {process.returncode}")
        raise RuntimeError(f"rsync failed with return code {process.returncode}")


def format_size(size_bytes):
    """Format bytes to human-readable size"""
    if size_bytes < 1024:
        return f"{size_bytes:.2f} B"
    elif size_bytes < 1024**2:
        return f"{size_bytes / 1024:.2f} KB"
    elif size_bytes < 1024**3:
        return f"{size_bytes / (1024 ** 2):.2f} MB"
    elif size_bytes < 1024**4:
        return f"{size_bytes / (1024 ** 3):.2f} GB"
    else:
        return f"{size_bytes / (1024 ** 4):.2f} TB"


def format_time(seconds):
    """Format seconds to human-readable time"""
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 3600:
        minutes = seconds // 60
        seconds %= 60
        return f"{int(minutes)} minutes {int(seconds)} seconds"
    else:
        hours = seconds // 3600
        seconds %= 3600
        minutes = seconds // 60
        seconds %= 60
        return f"{int(hours)} hours {int(minutes)} minutes {int(seconds)} seconds"
