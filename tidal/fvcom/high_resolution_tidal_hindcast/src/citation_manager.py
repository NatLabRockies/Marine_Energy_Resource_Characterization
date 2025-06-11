from pathlib import Path
import warnings
import re
from citeproc import (
    CitationStylesStyle,
    CitationStylesBibliography,
    Citation,
    CitationItem,
)
from citeproc.source.bibtex import BibTeX

REFERENCES_DIR = Path(Path(__file__).parent.parent, "references")

REFERENCES_FILE = Path(REFERENCES_DIR, "./references.bib")
# CSL_FILE = Path(REFERENCES_DIR, "./apa-6th-edition.csl")
# CSL_FILE = Path(REFERENCES_DIR, "./ieee-with-url.csl")
CSL_FILE = Path(REFERENCES_DIR, "./chicago-author-date.csl")


def clean_html_formatting(text):
    """
    Remove HTML formatting from citation text and convert smart quotes
    """
    # Remove HTML italic tags
    text = re.sub(r"<i>(.*?)</i>", r"\1", text)
    # Remove HTML bold tags (if any)
    text = re.sub(r"<b>(.*?)</b>", r"\1", text)
    # Convert HTML entities
    text = text.replace("&amp;", "&")
    text = text.replace("&lt;", "<")
    text = text.replace("&gt;", ">")
    text = text.replace("&quot;", '"')
    text = text.replace("&#39;", "'")

    # Convert Unicode smart quotes to straight quotes
    text = text.replace("\u201c", '"')  # Left double quotation mark
    text = text.replace("\u201d", '"')  # Right double quotation mark
    text = text.replace("\u2018", "'")  # Left single quotation mark
    text = text.replace("\u2019", "'")  # Right single quotation mark
    text = text.replace("\u2010", "-")  # Hyphen (from your output)

    text = text.replace(
        '"', "'"
    )  # Convert double quotes to single quotes (Helps with json output)

    return text


def format_reference(reference_id, show_warnings=False):
    """
    Format a single reference using modern bibtexparser and CSL
    Args:
        reference_id: the citation key to format
        show_warnings: if False (default), suppress citeproc warnings
    Returns:
        formatted reference string
    """
    # Conditionally silence warnings
    if not show_warnings:
        warnings.filterwarnings("ignore", category=UserWarning, module="citeproc")

    try:
        # For CSL processing, specify UTF-8 encoding
        bib_source = BibTeX(REFERENCES_FILE, encoding="utf-8")

        # Check if the reference exists
        if reference_id not in bib_source:
            return f"Reference '{reference_id}' not found"

        # Load CSL style
        style = CitationStylesStyle(CSL_FILE)

        # Create bibliography
        bibliography = CitationStylesBibliography(style, bib_source)

        # Create proper Citation object with CitationItem
        citation_item = CitationItem(reference_id)
        citation = Citation([citation_item])

        # Register the citation
        bibliography.register(citation)

        # Generate formatted reference
        for item in bibliography.bibliography():
            formatted_text = str(item)
            # Clean HTML formatting
            clean_text = clean_html_formatting(formatted_text)
            return clean_text

        return f"Reference '{reference_id}' formatting failed"

    except Exception as e:
        return f"Error formatting '{reference_id}': {str(e)}"
    finally:
        # Reset warning filters if we changed them
        if not show_warnings:
            warnings.resetwarnings()


def format_references(references, show_warnings=False):
    """
    Format multiple references using modern bibtexparser and CSL
    Args:
        references: list of citation keys to format
        show_warnings: if False (default), suppress citeproc warnings
    Returns:
        formatted references string
    """
    formatted_references = []
    for ref in references:
        formatted_references.append(format_reference(ref, show_warnings=show_warnings))
    # return "\n".join(formatted_references)
    return formatted_references


# Test the functions
if __name__ == "__main__":
    # Test with a single reference (show warnings for testing)
    print("Testing single reference:")
    result = format_reference("mhkdr_submission", show_warnings=True)
    print(result)
    print()

    # Test with multiple references (show warnings for testing)
    print("Testing multiple references:")
    test_refs = ["mhkdr_submission", "all_deb2024_tidal_iec"]
    results = format_references(test_refs, show_warnings=True)
    print(results)
