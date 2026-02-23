"""Generate the Variable Documentation section for tidal-hindcast.md from JSON spec.

Reads documentation_variable_spec.json (the single source of truth) and generates
the mkdocs-compatible markdown content between GENERATED:VARIABLE_DOCS_START and
GENERATED:VARIABLE_DOCS_END markers in docs/tidal-hindcast.md.

Usage:
    python generate_mkdocs_variable_section.py            # Write to tidal-hindcast.md
    python generate_mkdocs_variable_section.py --dry-run   # Preview without writing
"""

import json
import re
import argparse
from pathlib import Path


SPEC_PATH = Path(__file__).parent / "documentation_variable_spec.json"
DOCS_PATH = (
    Path(__file__).parent.parent.parent.parent / "docs" / "tidal-hindcast.md"
)

START_MARKER = "<!-- GENERATED:VARIABLE_DOCS_START -->"
END_MARKER = "<!-- GENERATED:VARIABLE_DOCS_END -->"


def load_spec():
    with open(SPEC_PATH) as f:
        return json.load(f)


def clean_description(text):
    """Clean up description text for markdown rendering.

    Fixes formatting issues from the source registry without modifying the
    registry data itself:
    - Capitalizes the first letter
    - Missing spaces after periods before capital letters
    - Missing spaces after periods before quotes/links
    - Normalizes paragraph breaks to proper markdown double-newlines
    """
    # Fix missing space after period before capital letter (e.g. "avoidance.Engineering")
    text = re.sub(r"\.([A-Z])", r". \1", text)
    # Fix missing space after period before opening quote or link
    text = re.sub(r'\.(["[\(])', r". \1", text)
    # Fix double periods that might result from the "is the <COMPLETE_DESCRIPTION>." pattern
    text = re.sub(r"\.\.", ".", text)
    # Normalize paragraph breaks - ensure \n\n becomes proper markdown paragraph breaks
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()
    # Capitalize first letter
    if text:
        text = text[0].upper() + text[1:]
    return text


def strip_outer_delimiters(equation):
    """Strip outer $...$ delimiters from an equation string."""
    eq = equation.strip()
    if eq.startswith("$") and eq.endswith("$"):
        eq = eq[1:-1].strip()
    return eq


def generate_quick_reference(spec):
    """Generate the Quick Reference table."""
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


def generate_variable_section(var):
    """Generate a single variable's documentation section."""
    display = var["display_name"]
    anchor = var["anchor"]
    col = var["column_name"]
    units = var["units"]
    one_liner = var["one_liner"]
    desc = clean_description(var["complete_description"])

    # Title includes units in brackets
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
        lines.extend([
            "",
            "**Equation**",
            "",
            "$$",
            eq_inner,
            "$$",
        ])

    if "equation_variables" in var:
        lines.extend([
            "",
            "**Where:**",
            "",
        ])
        for ev in var["equation_variables"]:
            lines.append(f"- {ev}")

    # Metadata table
    lines.extend([
        "",
        "| Property | Value |",
        "| --- | --- |",
        f"| Internal Name | `{col}` |",
        f"| Units | {units} |",
    ])

    lines.extend(["", "---"])
    return "\n".join(lines)


def generate_full_section(spec):
    """Generate the complete variable documentation section content."""
    parts = [generate_quick_reference(spec), "", "---", ""]

    for _col_name, var in spec.items():
        parts.append(generate_variable_section(var))
        parts.append("")

    return "\n".join(parts).rstrip() + "\n"


def inject_into_docs(content, dry_run=False):
    """Replace content between markers in tidal-hindcast.md."""
    doc_text = DOCS_PATH.read_text()

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

    DOCS_PATH.write_text(new_doc)
    print(f"Updated {DOCS_PATH}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate variable documentation for tidal-hindcast.md"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview generated content without writing to file",
    )
    args = parser.parse_args()

    spec = load_spec()
    content = generate_full_section(spec)
    inject_into_docs(content, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
