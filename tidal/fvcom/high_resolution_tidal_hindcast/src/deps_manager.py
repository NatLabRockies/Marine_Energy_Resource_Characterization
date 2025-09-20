"""
Dependency Manager

This module provides a centralized class for downloading and managing GIS dependencies
used by the marine energy resource characterization pipeline.
"""

import os
import zipfile
import urllib.request
from pathlib import Path
import time


class DependencyManager:
    """
    Manage downloading and extraction of GIS dependencies.

    This class handles downloading zip files from URLs and extracting them to
    the appropriate directories as specified in the configuration.
    """

    def __init__(self, config):
        """
        Initialize the dependency manager.

        Parameters:
            config (dict): Configuration dictionary containing dependencies and directories
        """
        self.config = config
        self.deps_dir = Path(config['dir']['dependencies'])

        # Create dependencies directory if it doesn't exist
        self.deps_dir.mkdir(parents=True, exist_ok=True)

    def download_dependency(self, dep_name):
        """
        Download and extract a specific dependency into its own subdirectory.

        Parameters:
            dep_name (str): Name of the dependency in config['dependencies']['gis']

        Returns:
            Path: Path to the extracted dependency directory

        Raises:
            ValueError: If dependency name is not found in config
            RuntimeError: If download or extraction fails
        """
        if dep_name not in self.config['dependencies']['gis']:
            raise ValueError(f"Dependency '{dep_name}' not found in configuration")

        dep_config = self.config['dependencies']['gis'][dep_name]

        # Determine file name from URL
        download_url = dep_config['data']
        filename = download_url.split('/')[-1]

        # Create a subdirectory for this dependency
        dep_subdir = self.deps_dir / dep_name
        dep_subdir.mkdir(parents=True, exist_ok=True)

        zip_path = dep_subdir / filename
        extract_dir = dep_subdir

        try:
            # Download if not present
            if not zip_path.exists():
                print(f"Downloading {dep_name} from {download_url}...")
                start_time = time.time()
                urllib.request.urlretrieve(download_url, zip_path)
                download_time = time.time() - start_time
                print(f"Downloaded {dep_name} to {zip_path} in {download_time:.2f} seconds")

            # Extract if contents don't exist (check for any .gpkg or .shp files in the directory)
            existing_geo_files = list(extract_dir.glob('**/*.gpkg')) + list(extract_dir.glob('**/*.shp'))
            if not existing_geo_files:
                print(f"Extracting {dep_name} to {extract_dir}...")
                start_time = time.time()
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
                extract_time = time.time() - start_time
                print(f"Extracted {dep_name} in {extract_time:.2f} seconds")

            return extract_dir

        except Exception as e:
            raise RuntimeError(f"Failed to download/extract {dep_name}: {str(e)}")

    def download_all_dependencies(self):
        """
        Download and extract all GIS dependencies.

        Returns:
            dict: Dictionary mapping dependency names to their extracted directory paths
        """
        print("Downloading all GIS dependencies...")
        start_time = time.time()

        extracted_paths = {}
        for dep_name in self.config['dependencies']['gis'].keys():
            extracted_paths[dep_name] = self.download_dependency(dep_name)

        total_time = time.time() - start_time
        print(f"All dependencies downloaded and extracted in {total_time:.2f} seconds")

        return extracted_paths

    def get_dependency_path(self, dep_name):
        """
        Get the path to an extracted dependency directory.

        Parameters:
            dep_name (str): Name of the dependency

        Returns:
            Path: Path to the dependency directory

        Raises:
            ValueError: If dependency name is not found
            FileNotFoundError: If dependency directory doesn't exist
        """
        if dep_name not in self.config['dependencies']['gis']:
            raise ValueError(f"Dependency '{dep_name}' not found in configuration")

        # Use the new subdirectory structure
        extract_dir = self.deps_dir / dep_name

        if not extract_dir.exists():
            raise FileNotFoundError(
                f"Dependency '{dep_name}' not found at {extract_dir}. "
                f"Run download_dependency('{dep_name}') first."
            )

        return extract_dir

    def is_dependency_available(self, dep_name):
        """
        Check if a dependency is downloaded and extracted.

        Parameters:
            dep_name (str): Name of the dependency

        Returns:
            bool: True if dependency is available, False otherwise
        """
        try:
            self.get_dependency_path(dep_name)
            return True
        except (ValueError, FileNotFoundError):
            return False

    def list_available_dependencies(self):
        """
        List all available (downloaded and extracted) dependencies.

        Returns:
            list: List of dependency names that are available
        """
        available = []
        for dep_name in self.config['dependencies']['gis'].keys():
            if self.is_dependency_available(dep_name):
                available.append(dep_name)
        return available

    def get_dependency_info(self, dep_name):
        """
        Get information about a dependency.

        Parameters:
            dep_name (str): Name of the dependency

        Returns:
            dict: Dictionary with dependency information
        """
        if dep_name not in self.config['dependencies']['gis']:
            raise ValueError(f"Dependency '{dep_name}' not found in configuration")

        dep_config = self.config['dependencies']['gis'][dep_name]

        return {
            'name': dep_name,
            'docs_url': dep_config['docs'],
            'data_url': dep_config['data'],
            'is_available': self.is_dependency_available(dep_name),
            'path': self.get_dependency_path(dep_name) if self.is_dependency_available(dep_name) else None
        }

    def find_gis_file(self, dep_name, pattern_name=None):
        """
        Find a GIS file within a dependency, preferring .gpkg over .shp files.

        Parameters:
            dep_name (str): Name of the dependency
            pattern_name (str, optional): Pattern to match in filename (case-insensitive)

        Returns:
            Path: Path to the GIS file

        Raises:
            FileNotFoundError: If no matching GIS file is found
        """
        dep_dir = self.get_dependency_path(dep_name)

        # Look for .gpkg files first (preferred)
        gpkg_files = list(dep_dir.glob('**/*.gpkg'))
        if pattern_name:
            gpkg_files = [f for f in gpkg_files if pattern_name.lower() in f.name.lower()]

        if gpkg_files:
            return gpkg_files[0]  # Return first match

        # Fall back to .shp files
        shp_files = list(dep_dir.glob('**/*.shp'))
        if pattern_name:
            shp_files = [f for f in shp_files if pattern_name.lower() in f.name.lower()]

        if shp_files:
            return shp_files[0]  # Return first match

        # No files found
        file_type = f"matching '{pattern_name}'" if pattern_name else "GIS files"
        raise FileNotFoundError(
            f"No {file_type} (.gpkg or .shp) found in dependency '{dep_name}' at {dep_dir}"
        )