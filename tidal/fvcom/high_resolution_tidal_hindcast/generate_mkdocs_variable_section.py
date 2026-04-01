"""Generate Variable Documentation for MkDocs from JSON spec.

Reads documentation_variable_spec.json (the single source of truth) and generates
mkdocs-compatible markdown content in one of two modes:

  - inline: Inject into a single file between GENERATED markers (legacy behavior)
  - pages:  Write individual variable .md files + a quick reference index.md

Usage:
    python generate_mkdocs_variable_section.py                      # inline mode (legacy)
    python generate_mkdocs_variable_section.py --mode pages         # multi-file mode
    python generate_mkdocs_variable_section.py --mode pages --dry-run
"""

import json
import re
import argparse
from pathlib import Path


SPEC_PATH = Path(__file__).parent / "documentation_variable_spec.json"
DOCS_ROOT = Path(__file__).parent.parent.parent.parent / "docs"

# Legacy inline target
INLINE_DOCS_PATH = DOCS_ROOT / "tidal-hindcast.md"
START_MARKER = "<!-- GENERATED:VARIABLE_DOCS_START -->"
END_MARKER = "<!-- GENERATED:VARIABLE_DOCS_END -->"

# Multi-page target
PAGES_DIR = DOCS_ROOT / "tidal" / "high_resolution_hindcast" / "variables"

# Auto-generated file header (HTML comment, invisible in rendered output)
AUTOGEN_HEADER = (
    "<!-- AUTO-GENERATED FILE — DO NOT EDIT DIRECTLY -->\n"
    "<!-- Source of truth: src/variable_registry.py (VARIABLE_REGISTRY) -->\n"
    "<!-- To update: edit the registry, then run `python generate_variable_docs.py` -->\n"
)

# Snippet include appended to every page
CITE_SNIPPET = (
    '\n--8<-- "docs/tidal/high_resolution_hindcast/_cite-widget.md"\n'
)


def load_spec():
    with open(SPEC_PATH) as f:
        return json.load(f)


def clean_description(text):
    """Clean up description text for markdown rendering."""
    text = re.sub(r"\.([A-Z])", r". \1", text)
    text = re.sub(r'\.(["[\(])', r". \1", text)
    text = re.sub(r"\.\.", ".", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()
    if text:
        text = text[0].upper() + text[1:]
    return text


def strip_outer_delimiters(equation):
    """Strip outer $...$ delimiters from an equation string."""
    eq = equation.strip()
    if eq.startswith("$") and eq.endswith("$"):
        eq = eq[1:-1].strip()
    return eq


# ---------------------------------------------------------------------------
# Inline mode (legacy) — inject into a single file between markers
# ---------------------------------------------------------------------------


def generate_quick_reference_inline(spec):
    """Generate the Quick Reference table with anchor links (inline mode)."""
    lines = [
        "### Variable Quick Reference",
        "",
        "| Variable | Internal Name | Units | Description |",
        "| --- | --- | --- | --- |",
    ]
    for _col_name, var in spec.items():
        display = var["display_name"]
        anchor = var["anchor"]
        col = var["column_name"]
        units = var["units"]
        one_liner = var["one_liner"]
        lines.append(
            f"| [{display}](#{anchor}) | `{col}` | {units} | {one_liner} |"
        )
    return "\n".join(lines)


def generate_variable_section_inline(var):
    """Generate a single variable's documentation section (inline mode)."""
    display = var["display_name"]
    anchor = var["anchor"]
    col = var["column_name"]
    units = var["units"]
    one_liner = var["one_liner"]
    desc = clean_description(var["complete_description"])

    units_part = f" [{units}]" if units else ""
    lines = [
        f"### {display}{units_part} {{#{anchor}}}",
        "",
        f"*{one_liner}*",
        "",
        "**Description**",
        "",
        desc,
    ]

    if "equation" in var:
        eq_inner = strip_outer_delimiters(var["equation"])
        lines.extend(["", "**Equation**", "", "$$", eq_inner, "$$"])

    if "equation_variables" in var:
        lines.extend(["", "**Where:**", ""])
        for ev in var["equation_variables"]:
            lines.append(f"- {ev}")

    lines.extend([
        "",
        "| Property | Value |",
        "| --- | --- |",
        f"| Internal Name | `{col}` |",
        f"| Units | {units} |",
        "",
        "---",
    ])
    return "\n".join(lines)


def generate_full_section_inline(spec):
    """Generate the complete variable documentation section (inline mode)."""
    parts = [generate_quick_reference_inline(spec), "", "---", ""]
    for _col_name, var in spec.items():
        parts.append(generate_variable_section_inline(var))
        parts.append("")
    return "\n".join(parts).rstrip() + "\n"


def inject_into_docs(content, dry_run=False):
    """Replace content between markers in tidal-hindcast.md."""
    doc_text = INLINE_DOCS_PATH.read_text()
    start_idx = doc_text.find(START_MARKER)
    end_idx = doc_text.find(END_MARKER)

    if start_idx == -1:
        raise ValueError(f"Start marker not found: {START_MARKER}")
    if end_idx == -1:
        raise ValueError(f"End marker not found: {END_MARKER}")

    new_doc = (
        doc_text[: start_idx + len(START_MARKER)]
        + "\n"
        + content
        + END_MARKER
        + doc_text[end_idx + len(END_MARKER) :]
    )

    if dry_run:
        print(content)
        return

    INLINE_DOCS_PATH.write_text(new_doc)
    print(f"Updated {INLINE_DOCS_PATH}")


# ---------------------------------------------------------------------------
# Pages mode — write individual files
# ---------------------------------------------------------------------------


def generate_variable_page(var):
    """Generate a standalone markdown page for a single variable."""
    display = var["display_name"]
    col = var["column_name"]
    units = var["units"]
    one_liner = var["one_liner"]
    desc = clean_description(var["complete_description"])

    units_part = f" [{units}]" if units else ""
    lines = [
        f"# {display}{units_part}",
        "",
        f"*{one_liner}*",
        "",
        "## Description",
        "",
        desc,
    ]

    if "equation" in var:
        eq_inner = strip_outer_delimiters(var["equation"])
        lines.extend(["", "## Equation", "", "$$", eq_inner, "$$"])

    if "equation_variables" in var:
        lines.extend(["", "**Where:**", ""])
        for ev in var["equation_variables"]:
            lines.append(f"- {ev}")

    lines.extend([
        "",
        "## Properties",
        "",
        "| Property | Value |",
        "| --- | --- |",
        f"| Internal Name | `{col}` |",
        f"| Units | {units} |",
    ])

    return "\n".join(lines) + "\n"


def generate_quick_reference_page(spec):
    """Generate the index.md quick reference page with links to individual pages."""
    lines = [
        "# Variable Quick Reference",
        "",
        "Summary of all variables in the High Resolution Tidal Hindcast dataset. "
        "Click a variable name to view its full documentation including description, "
        "equation, and properties.",
        "",
        "| Variable | Internal Name | Units | Description |",
        "| --- | --- | --- | --- |",
    ]
    for _col_name, var in spec.items():
        display = var["display_name"]
        anchor = var["anchor"]
        col = var["column_name"]
        units = var["units"]
        one_liner = var["one_liner"]
        lines.append(
            f"| [{display}]({anchor}.md) | `{col}` | {units} | {one_liner} |"
        )
    return "\n".join(lines) + "\n"


def generate_all_pages(spec):
    """Return {filename: markdown_content} for all variable pages + index.

    Each page includes the auto-generated header and cite-widget snippet.
    Content is generated in memory — nothing is written to disk.
    """
    pages = {}

    # Index page
    pages["index.md"] = AUTOGEN_HEADER + "\n" + generate_quick_reference_page(spec) + CITE_SNIPPET

    # Individual variable pages
    for _col_name, var in spec.items():
        anchor = var["anchor"]
        filename = f"{anchor}.md"
        pages[filename] = AUTOGEN_HEADER + "\n" + generate_variable_page(var) + CITE_SNIPPET

    return pages


def write_pages(spec, dry_run=False):
    """Write individual variable pages and the quick reference index."""
    pages = generate_all_pages(spec)

    if not dry_run:
        PAGES_DIR.mkdir(parents=True, exist_ok=True)

    for filename, content in pages.items():
        page_path = PAGES_DIR / filename
        if dry_run:
            print(f"--- {page_path} ---")
            print(content)
        else:
            page_path.write_text(content)
            print(f"Wrote {page_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Generate variable documentation for MkDocs"
    )
    parser.add_argument(
        "--mode",
        choices=["inline", "pages"],
        default="inline",
        help="Output mode: 'inline' injects into tidal-hindcast.md (legacy), "
        "'pages' writes individual files to docs/tidal/high_resolution_hindcast/variables/",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview generated content without writing to file(s)",
    )
    args = parser.parse_args()

    spec = load_spec()

    if args.mode == "inline":
        content = generate_full_section_inline(spec)
        inject_into_docs(content, dry_run=args.dry_run)
    elif args.mode == "pages":
        write_pages(spec, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
