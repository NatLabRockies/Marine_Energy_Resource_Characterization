# Quality Assurance

## Verification Process

| Category             | Verification Process                                | Status    |
| -------------------- | --------------------------------------------------- | --------- |
| Model Specification  | FVCOM version, CF conventions, required variables   | Compliant |
| Temporal Integrity   | Time steps, chronological order, expected frequency | Compliant |
| Spatial Consistency  | Coordinate systems, grid consistency                | Compliant |
| Metadata Consistency | Global attributes across files                      | Compliant |
| Dataset Structure    | Dimensions, variables, coordinates                  | Compliant |
| Completeness         | Temporal coverage, required variables               | Compliant |

## Known Data Gaps

| Location         | Issue                                     |
| ---------------- | ----------------------------------------- |
| Puget Sound      | Missing 2015-12-31                        |
| Aleutian Islands | File MD_AIS_west_hrBathy_0370.nc excluded |
| Cook Inlet       | File cki_0366.nc excluded                 |
| Piscataqua River | File PIR_0368.nc excluded                 |

## Temporal Validation

Each location's time series was verified for:

- **Completeness** — All expected timesteps present (within known gaps above)
- **Chronological order** — Timestamps monotonically increasing
- **Frequency consistency** — Uniform hourly or half-hourly intervals
- **No duplicates** — Each timestep appears exactly once

## Spatial Validation

Grid consistency was verified for:

- **Coordinate range** — All coordinates within expected geographic bounds
- **Grid connectivity** — Triangular element definitions consistent across timesteps
- **Projection accuracy** — UTM-to-WGS84 transformations validated (see [Coordinate Systems](coordinate-systems.md))

## Value Range Checks

Derived variables were checked against physically reasonable bounds:

- **Current speed** — Non-negative, within expected range for each location
- **Power density** — Non-negative, consistent with cubic relationship to speed
- **Water depth** — Positive in non-dry cells, consistent with bathymetry + elevation
- **Tidal range** — Positive, within expected range for tidal regime classification


--8<-- "docs/tidal/high_resolution_hindcast/_cite-widget.md"
