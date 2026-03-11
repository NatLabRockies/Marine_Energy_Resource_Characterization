"""MkDocs hook: verify variable docs are in sync with the registry.

This on_pre_build hook generates the expected documentation content in memory
and compares it against what's on disk.  If anything is stale the build fails
with a clear message telling the developer how to fix it.

Registered in mkdocs.yml:
    hooks:
      - tidal/fvcom/high_resolution_tidal_hindcast/hooks/gen_variable_docs.py
"""

import json
import logging
import sys
from pathlib import Path

log = logging.getLogger("mkdocs.hooks.gen_variable_docs")

# Resolve paths relative to this file's location
_HOOK_DIR = Path(__file__).resolve().parent        # hooks/
_PROJECT_DIR = _HOOK_DIR.parent                     # high_resolution_tidal_hindcast/
_REPO_ROOT = _PROJECT_DIR.parent.parent.parent      # marine_energy_resource_characterization/


def on_pre_build(config, **kwargs):
    """Verify variable docs are in sync with registry. Fail build if stale."""

    # Make project dir importable
    if str(_PROJECT_DIR) not in sys.path:
        sys.path.insert(0, str(_PROJECT_DIR))

    # Import the registry (the single source of truth)
    from src.variable_registry import documentation_variable_specification

    # Import the generator functions (never writes to disk)
    from generate_mkdocs_variable_section import generate_all_pages

    # --- Check JSON spec file ---
    doc_spec_path = _PROJECT_DIR / "documentation_variable_spec.json"
    expected_json = json.dumps(documentation_variable_specification, indent=2) + "\n"

    stale_files = []

    if not doc_spec_path.exists():
        stale_files.append(str(doc_spec_path.relative_to(_REPO_ROOT)))
    elif doc_spec_path.read_text() != expected_json:
        stale_files.append(str(doc_spec_path.relative_to(_REPO_ROOT)))

    # --- Check MD pages ---
    pages_dir = _REPO_ROOT / "docs" / "tidal" / "high_resolution_hindcast" / "variables"
    expected_pages = generate_all_pages(documentation_variable_specification)

    for filename, expected_content in expected_pages.items():
        page_path = pages_dir / filename
        if not page_path.exists():
            stale_files.append(str(page_path.relative_to(_REPO_ROOT)))
        elif page_path.read_text() != expected_content:
            stale_files.append(str(page_path.relative_to(_REPO_ROOT)))

    if stale_files:
        file_list = "\n".join(f"  - {f}" for f in stale_files)
        raise SystemExit(
            "\n"
            "ERROR: Variable documentation is out of sync with the registry.\n"
            "\n"
            "Stale files:\n"
            f"{file_list}\n"
            "\n"
            "To fix, run:\n"
            "  cd tidal/fvcom/high_resolution_tidal_hindcast && python generate_variable_docs.py\n"
            "\n"
            "To discard manual MD edits:\n"
            "  git checkout -- docs/tidal/high_resolution_hindcast/variables/\n"
        )

    log.info("Variable docs up to date")
