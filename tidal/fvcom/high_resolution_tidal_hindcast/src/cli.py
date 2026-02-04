"""Command line interface for FVCOM data processing pipeline.

The CLI supports two main arguments:
    - location: The geographic location of the FVCOM dataset to process
    - output-type: The type of processing to perform (summary, std, or all)

Example:
    To use in a script:
        from src.cli import parse_args
        args = parse_args(config)
        location_config = config["locations"][args.location]

    Command line usage:
        python runner.py aleutian_islands
        python runner.py cook_inlet --output-type std
"""

import argparse


def validate_location(config):
    """Create a validation function closure with access to config."""

    def _validate_location(location):
        if location not in config["location_specification"]:
            valid_locations = list(config["location_specification"].keys())
            raise argparse.ArgumentTypeError(
                f"Invalid location: {location}. Must be one of: {', '.join(valid_locations)}"
            )
        return location

    return _validate_location


def validate_output_type(output_type):
    """Validate output type and return standardized value."""
    valid_types = ["summary", "std", "all"]
    output_type = output_type.lower()
    if output_type not in valid_types:
        raise argparse.ArgumentTypeError(
            f"Invalid output type: {output_type}. Must be one of: {', '.join(valid_types)}"
        )
    return output_type


def parse_args(config):
    """Parse command line arguments using provided config."""
    parser = argparse.ArgumentParser(
        description="Process FVCOM data for specified location and output type."
    )

    parser.add_argument(
        "location",
        type=validate_location(config),
        help="Location to process (e.g., aleutian_islands, cook_inlet)",
    )

    parser.add_argument(
        "--output-type",
        type=validate_output_type,
        default="all",
        help="Output type to generate (summary, std, or all). Defaults to all.",
    )

    return parser.parse_args()


def parse_partition_args(config):
    """Parse command line arguments using provided config."""
    parser = argparse.ArgumentParser(
        description="Process FVCOM data for specified location and output type."
    )
    parser.add_argument(
        "location",
        type=validate_location(config),
        help="Location to process (e.g., aleutian_islands, cook_inlet)",
    )
    # Add batch size argument
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of items to process in each batch (default: 100)",
    )
    # Add batch number argument
    parser.add_argument(
        "--batch-num", type=int, default=1, help="Batch number to process (default: 1)"
    )
    # Add skip existing argument
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=False,
        help="Skip processing faces whose output files already exist",
    )

    return parser.parse_args()
