# Contributing

Guide for contributing to the Marine Energy Resource Characterization documentation.

## Documentation Setup

### Prerequisites

- Python 3.11+
- pip

### Installation

Install MkDocs with the Material theme:

```bash
pip install mkdocs-material
```

### Local Development

Serve the documentation locally with live reload:

```bash
cd /path/to/marine_energy_resource_characterization
mkdocs serve
```

The site will be available at [http://127.0.0.1:8000](http://127.0.0.1:8000). Changes to markdown files will automatically reload.

### Build

Build the static site:

```bash
mkdocs build
```

Output is written to the `site/` directory.

### Build with Strict Mode

Check for warnings and errors:

```bash
mkdocs build --strict
```

## Documentation Structure

```
docs/
├── index.md              # Home page
├── tidal-hindcast.md     # Tidal hindcast documentation
├── wave-hindcast.md      # Wave hindcast documentation
├── stylesheets/
│   └── extra.css         # NLR color overrides
└── javascripts/
    └── mathjax.js        # LaTeX equation support

mkdocs.yml                # MkDocs configuration
```

## Writing Guidelines

### Math Equations

Use LaTeX syntax for equations. Inline math uses single dollar signs:

```markdown
The velocity magnitude is $U = \sqrt{u^2 + v^2}$.
```

Display equations use double dollar signs:

```markdown
$$
\bar{\bar{U}} = \text{mean}\left(\left[\text{mean}(U_{1,t}, \ldots, U_{N_\sigma,t})\right]\right)
$$
```

### Admonitions

Use MkDocs Material admonitions for callouts:

```markdown
!!! info "Title"
Information content here.

!!! warning "Title"
Warning content here.

!!! tip "Title"
Tip content here.
```

### Anchor Links

Create linkable sections with explicit anchors:

```markdown
### Section Name {#section-anchor}
```

Link to anchors:

```markdown
See [Section Name](#section-anchor) for details.
```

### Tables

Use standard markdown tables:

```markdown
| Column 1 | Column 2 |
| -------- | -------- |
| Value 1  | Value 2  |
```

## Generating Variable Documentation

The tidal hindcast variable documentation can be regenerated from source:

```bash
cd tidal/fvcom/high_resolution_tidal_hindcast
python generate_atlas_variable_docs.py --format markdown
```

Options:

- `--format docx` - Generate Word document (default)
- `--format markdown` - Generate MkDocs-compatible markdown
- `--output PATH` - Custom output path

## Deployment

Documentation is automatically deployed to GitHub Pages when changes are pushed to the `main` branch. The GitHub Actions workflow (`.github/workflows/deploy-docs.yml`) handles the build and deployment.

To manually deploy:

```bash
mkdocs gh-deploy --force
```

## NLR Branding

The documentation uses NLR brand colors defined in `docs/stylesheets/extra.css`:

| Color          | Hex       | Usage                 |
| -------------- | --------- | --------------------- |
| Primary Blue   | `#0079C2` | Headers, links        |
| Secondary Blue | `#00A3E4` | Hover states, accents |
| Yellow         | `#FFC423` | Warnings              |
| Green          | `#7DA544` | Success states        |
| Gray           | `#626D72` | Footer, text          |

## Questions

For questions about the documentation, open an issue on GitHub or contact [marineresource@nlr.gov](mailto:marineresource@nlr.gov).
