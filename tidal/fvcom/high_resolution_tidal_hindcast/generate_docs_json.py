"""Generate atlas and documentation variable specification JSON files."""

import json

from src.variable_registry import (
    atlas_variable_specification,
    documentation_variable_specification,
)

with open("atlas_variable_spec.json", "w") as f:
    json.dump(atlas_variable_specification, f, indent=2)

with open("documentation_variable_spec.json", "w") as f:
    json.dump(documentation_variable_specification, f, indent=2)

print(f"Wrote atlas_variable_spec.json ({len(atlas_variable_specification)} vars)")
print(f"Wrote documentation_variable_spec.json ({len(documentation_variable_specification)} vars)")
