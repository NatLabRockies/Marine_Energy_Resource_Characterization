#!/usr/bin/env python3
"""Build mkdocs documentation for PR preview deployment.

This script builds mkdocs with a modified site_url to ensure relative paths
work correctly in PR preview subdirectories.

Usage:
    python scripts/build_pr_preview.py <pr_number> <output_dir>
"""

import sys
import yaml
import tempfile
import subprocess
from pathlib import Path


def build_pr_preview(pr_number, output_dir):
    """Build docs for PR preview with adjusted site_url."""
    
    repo_name = "Marine_Energy_Resource_Characterization"
    repo_url = "https://natlabrockies.github.io"
    pr_preview_url = f"{repo_url}/{repo_name}/pr-preview/pr-{pr_number}/"
    
    # Read mkdocs config
    config_path = Path(__file__).parent.parent / "mkdocs.yml"
    original_dir = config_path.parent
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Modify site_url for PR preview
    config["site_url"] = pr_preview_url
    
    # Write temporary config in the repo root so relative paths work
    temp_config_path = original_dir / ".mkdocs.pr-preview.yml"
    with open(temp_config_path, "w") as f:
        yaml.dump(config, f)
    
    try:
        # Build with modified config from repo root
        print(f"Building PR preview with site_url: {pr_preview_url}")
        result = subprocess.run(
            ["mkdocs", "build", "--config-file", str(temp_config_path), "--site-dir", output_dir],
            cwd=str(original_dir),
            check=True,
        )
        print(f"Build completed successfully to {output_dir}")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"Build failed: {e}", file=sys.stderr)
        return 1
    finally:
        # Clean up temp config
        temp_config_path.unlink(missing_ok=True)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: build_pr_preview.py <pr_number> <output_dir>", file=sys.stderr)
        sys.exit(1)
    
    pr_number = sys.argv[1]
    output_dir = sys.argv[2]
    
    sys.exit(build_pr_preview(pr_number, output_dir))
