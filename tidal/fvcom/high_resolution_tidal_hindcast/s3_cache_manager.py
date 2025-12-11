"""
S3 Cache Manager with ETag-based validation.

Provides lazy loading and caching of S3 objects with ETag verification
to avoid redundant downloads. Cached files are stored in a local directory
structure that mirrors S3 paths.

Usage:
    cache = S3CacheManager(
        bucket="oedi-data-drop",
        prefix="us-tidal",
        aws_profile="us-tidal"
    )

    # Get a file (downloads if not cached or ETag changed)
    local_path = cache.get("manifest/v1.0.0/manifest_1.0.0.json")

    # Get grid file
    grid_path = cache.get("manifest/v1.0.0/grids/lat_60/lon_-151/60_-151_72_42.json")
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Optional, Dict, Any

import boto3
from botocore.exceptions import ClientError


class S3CacheManager:
    """
    Manages local caching of S3 objects with ETag-based validation.

    Files are cached in a local directory structure that mirrors S3 paths.
    ETags are stored alongside cached files to detect when S3 objects change.
    """

    def __init__(
        self,
        bucket: str,
        prefix: str,
        cache_dir: Optional[Path] = None,
        aws_profile: Optional[str] = None,
    ):
        """
        Initialize S3 cache manager.

        Parameters
        ----------
        bucket : str
            S3 bucket name
        prefix : str
            S3 prefix (e.g., 'us-tidal')
        cache_dir : Path, optional
            Local cache directory. Defaults to ./us_tidal_cache
        aws_profile : str, optional
            AWS profile name for S3 access
        """
        self.bucket = bucket
        self.prefix = prefix
        self.cache_dir = cache_dir or Path("./us_tidal_cache")
        self.aws_profile = aws_profile

        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # ETag cache file (stores ETags for all cached files)
        self.etag_cache_file = self.cache_dir / ".etag_cache.json"
        self.etag_cache = self._load_etag_cache()

        # Initialize S3 client
        self._s3_client = None

    @property
    def s3(self):
        """Lazy-load S3 client."""
        if self._s3_client is None:
            if self.aws_profile:
                session = boto3.Session(profile_name=self.aws_profile)
                self._s3_client = session.client("s3")
            else:
                self._s3_client = boto3.client("s3")
        return self._s3_client

    def _load_etag_cache(self) -> Dict[str, str]:
        """Load ETag cache from disk."""
        if self.etag_cache_file.exists():
            try:
                with open(self.etag_cache_file, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return {}
        return {}

    def _save_etag_cache(self):
        """Save ETag cache to disk."""
        with open(self.etag_cache_file, "w") as f:
            json.dump(self.etag_cache, f, indent=2)

    def _get_s3_key(self, relative_path: str) -> str:
        """Convert relative path to full S3 key."""
        return f"{self.prefix}/{relative_path}"

    def _get_local_path(self, relative_path: str) -> Path:
        """Convert relative path to local cache path."""
        return self.cache_dir / relative_path

    def _get_s3_etag(self, s3_key: str) -> Optional[str]:
        """
        Get ETag for an S3 object.

        Returns None if object doesn't exist or error occurs.
        """
        try:
            response = self.s3.head_object(Bucket=self.bucket, Key=s3_key)
            return response.get("ETag", "").strip('"')
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                return None
            raise

    def _get_local_etag(self, relative_path: str) -> Optional[str]:
        """Get cached ETag for a relative path."""
        cache_key = f"{self.bucket}/{self.prefix}/{relative_path}"
        return self.etag_cache.get(cache_key)

    def _set_local_etag(self, relative_path: str, etag: str):
        """Store ETag for a relative path."""
        cache_key = f"{self.bucket}/{self.prefix}/{relative_path}"
        self.etag_cache[cache_key] = etag
        self._save_etag_cache()

    def is_cached(self, relative_path: str) -> bool:
        """
        Check if a file is cached locally.

        Parameters
        ----------
        relative_path : str
            Path relative to S3 prefix

        Returns
        -------
        bool
            True if file exists in cache
        """
        local_path = self._get_local_path(relative_path)
        return local_path.exists()

    def is_valid(self, relative_path: str) -> bool:
        """
        Check if cached file matches S3 ETag.

        Parameters
        ----------
        relative_path : str
            Path relative to S3 prefix

        Returns
        -------
        bool
            True if cached file exists and ETag matches S3
        """
        if not self.is_cached(relative_path):
            return False

        s3_key = self._get_s3_key(relative_path)
        s3_etag = self._get_s3_etag(s3_key)
        local_etag = self._get_local_etag(relative_path)

        return s3_etag is not None and s3_etag == local_etag

    def get(
        self,
        relative_path: str,
        force_download: bool = False,
        validate: bool = True,
    ) -> Path:
        """
        Get a file from cache, downloading from S3 if necessary.

        Parameters
        ----------
        relative_path : str
            Path relative to S3 prefix (e.g., "manifest/v1.0.0/manifest_1.0.0.json")
        force_download : bool, default False
            If True, always download from S3 even if cached
        validate : bool, default True
            If True, check ETag before using cached file.
            If False, use cached file without validation (faster but may be stale).

        Returns
        -------
        Path
            Local path to the cached file

        Raises
        ------
        FileNotFoundError
            If file doesn't exist on S3
        """
        local_path = self._get_local_path(relative_path)
        s3_key = self._get_s3_key(relative_path)

        # Check if we need to download
        need_download = force_download or not local_path.exists()

        if not need_download and validate:
            # Check if ETag matches
            s3_etag = self._get_s3_etag(s3_key)
            if s3_etag is None:
                raise FileNotFoundError(
                    f"S3 object not found: s3://{self.bucket}/{s3_key}"
                )

            local_etag = self._get_local_etag(relative_path)
            if s3_etag != local_etag:
                need_download = True

        if need_download:
            # Download from S3
            local_path.parent.mkdir(parents=True, exist_ok=True)

            try:
                self.s3.download_file(self.bucket, s3_key, str(local_path))
            except ClientError as e:
                if e.response["Error"]["Code"] == "404":
                    raise FileNotFoundError(
                        f"S3 object not found: s3://{self.bucket}/{s3_key}"
                    )
                raise

            # Store ETag
            s3_etag = self._get_s3_etag(s3_key)
            if s3_etag:
                self._set_local_etag(relative_path, s3_etag)

        return local_path

    def get_json(
        self,
        relative_path: str,
        force_download: bool = False,
        validate: bool = True,
    ) -> Dict[str, Any]:
        """
        Get a JSON file from cache and parse it.

        Parameters
        ----------
        relative_path : str
            Path relative to S3 prefix
        force_download : bool, default False
            If True, always download from S3
        validate : bool, default True
            If True, check ETag before using cached file

        Returns
        -------
        dict
            Parsed JSON content
        """
        local_path = self.get(relative_path, force_download, validate)
        with open(local_path, "r") as f:
            return json.load(f)

    def clear_cache(self, relative_path: Optional[str] = None):
        """
        Clear cached files.

        Parameters
        ----------
        relative_path : str, optional
            If specified, clear only this file. Otherwise clear entire cache.
        """
        if relative_path:
            local_path = self._get_local_path(relative_path)
            if local_path.exists():
                local_path.unlink()

            cache_key = f"{self.bucket}/{self.prefix}/{relative_path}"
            if cache_key in self.etag_cache:
                del self.etag_cache[cache_key]
                self._save_etag_cache()
        else:
            # Clear entire cache
            import shutil

            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.etag_cache = {}
            self._save_etag_cache()

    def cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns
        -------
        dict
            Cache statistics including file count, size, etc.
        """
        total_files = 0
        total_size = 0

        for root, dirs, files in os.walk(self.cache_dir):
            for f in files:
                if f.startswith("."):
                    continue
                total_files += 1
                total_size += (Path(root) / f).stat().st_size

        return {
            "cache_dir": str(self.cache_dir),
            "total_files": total_files,
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024 * 1024),
            "etag_entries": len(self.etag_cache),
        }
