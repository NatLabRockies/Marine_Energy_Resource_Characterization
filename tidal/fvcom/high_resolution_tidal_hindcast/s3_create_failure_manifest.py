#!/usr/bin/env python3
"""
Create JSON manifest of failed uploads from SQLite database.

This script runs after all upload jobs complete (via SLURM dependency)
and creates a JSON file listing all files that failed to upload.
"""

import argparse
import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path


def get_failed_files(manifest_path):
    """
    Query SQLite manifest for failed uploads.

    Args:
        manifest_path: Path to SQLite manifest file

    Returns:
        List of failed file records
    """
    conn = sqlite3.connect(manifest_path)
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT file_index, local_path, relative_path, s3_destination,
               file_size_bytes, file_extension, upload_timestamp
        FROM files
        WHERE upload_status = 'failed'
        ORDER BY file_index
        """
    )

    failed_files = cursor.fetchall()
    conn.close()

    return failed_files


def get_upload_summary(manifest_path):
    """
    Get summary statistics from manifest.

    Args:
        manifest_path: Path to SQLite manifest file

    Returns:
        Dict with counts by status
    """
    conn = sqlite3.connect(manifest_path)
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT upload_status, COUNT(*) as count
        FROM files
        GROUP BY upload_status
        """
    )

    summary = {row[0]: row[1] for row in cursor.fetchall()}
    conn.close()

    return summary


def create_failure_manifest(manifest_path):
    """
    Create JSON file with failed uploads.

    Args:
        manifest_path: Path to SQLite manifest file

    Returns:
        Path to created JSON file or None if no failures
    """
    print(f"Analyzing manifest: {manifest_path}")

    # Get upload summary
    summary = get_upload_summary(manifest_path)
    total = sum(summary.values())
    completed = summary.get("completed", 0)
    failed = summary.get("failed", 0)
    pending = summary.get("pending", 0)

    print("\nUpload Summary:")
    print(f"  Total files: {total}")
    print(f"  Completed:   {completed}")
    print(f"  Failed:      {failed}")
    print(f"  Pending:     {pending}")

    # Get failed files
    if failed == 0:
        print("\nNo failures detected - all uploads successful!")
        return None

    failed_files = get_failed_files(manifest_path)

    # Create failure manifest
    manifest_name = Path(manifest_path).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    failure_file = (
        Path(manifest_path).parent / f"failures_{manifest_name}_{timestamp}.json"
    )

    failure_data = {
        "manifest_path": str(manifest_path),
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total_files": total,
            "completed": completed,
            "failed": failed,
            "pending": pending,
        },
        "failed_files": [
            {
                "file_index": file_index,
                "local_path": local_path,
                "relative_path": relative_path,
                "s3_destination": s3_destination,
                "file_size_bytes": file_size_bytes,
                "file_extension": file_extension,
                "upload_timestamp": upload_timestamp,
            }
            for (
                file_index,
                local_path,
                relative_path,
                s3_destination,
                file_size_bytes,
                file_extension,
                upload_timestamp,
            ) in failed_files
        ],
    }

    with open(failure_file, "w") as f:
        json.dump(failure_data, f, indent=2)

    print(f"\nFailure manifest created: {failure_file}")
    print(f"Failed files: {failed}")

    return str(failure_file)


def main():
    parser = argparse.ArgumentParser(
        description="Create JSON manifest of failed uploads"
    )

    parser.add_argument("manifest_path", help="Path to SQLite manifest file")

    args = parser.parse_args()

    # Validate manifest exists
    if not Path(args.manifest_path).exists():
        print(f"Error: Manifest file not found: {args.manifest_path}", file=sys.stderr)
        sys.exit(1)

    # Create failure manifest
    failure_file = create_failure_manifest(args.manifest_path)

    if failure_file:
        print(f"\n{failure_file}")
        sys.exit(1)  # Exit with error if there were failures
    else:
        sys.exit(0)  # Exit success if no failures


if __name__ == "__main__":
    main()
