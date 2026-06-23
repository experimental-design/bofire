# BoFire Documentation with Quarto

This directory contains the unified documentation for BoFire, built using [Quarto](https://quarto.org/).

## Structure

The documentation is organized as follows:

```
docs/
├── index.qmd                    # Main landing page
├── install.qmd                  # Installation instructions
├── getting_started.ipynb        # Getting started tutorial
├── tutorials/                   # Tutorial notebooks organized by category
│   ├── basic_examples/
│   ├── advanced_examples/
│   ├── benchmarks/
│   ├── doe/
│   └── serialization/
├── userguides/                  # User guides for key concepts
│   ├── domain.qmd
│   ├── strategies.qmd
│   ├── surrogates.qmd
│   └── data_models_functionals.qmd
└── reference/                   # API reference documentation
    ├── domain.qmd
    ├── constraints.qmd
    ├── objectives.qmd
    ├── features.qmd
    └── utils.qmd
```

## Prerequisites

To build the documentation, you need:

1. **Quarto**: Install from [quarto.org](https://quarto.org/docs/get-started/)
   ```bash
   # On Linux
   wget https://github.com/quarto-dev/quarto-cli/releases/download/v1.4.549/quarto-1.4.549-linux-amd64.deb
   sudo dpkg -i quarto-1.4.549-linux-amd64.deb

   # On macOS
   brew install quarto

   # On Windows
   # Download and run the installer from quarto.org
   ```

2. **Python environment**: The BoFire package should be installed with all dependencies
   ```bash
   pip install -e .
   ```

3. **Jupyter**: Required for rendering notebooks
   ```bash
   pip install jupyter nbformat nbclient jupyter-cache
   ```

## Building the Documentation

### Preview During Development

To preview the documentation locally with live reload:

```bash
quarto preview
```

This will:
- Start a local web server (typically at http://localhost:4200)
- Automatically rebuild when you save changes
- Open the documentation in your browser

### Build the Static Site

To build the complete static site:

```bash
quarto render
```

The built site will be in the `_site/` directory.

### Build Specific Files

To render individual files:

```bash
quarto render docs/getting_started.ipynb
quarto render docs/index.qmd
```

## Publishing

The documentation is automatically published to GitHub Pages when changes are pushed to the main branch.

To manually publish:

```bash
quarto publish gh-pages
```

## Configuration

The documentation configuration is in `_quarto.yml` at the project root. Key settings include:

- **Navigation**: Navbar and sidebar structure
- **Theme**: Light/dark mode with custom SCSS
- **Execution**: Jupyter notebook execution settings
- **Output**: HTML formatting options

## Writing Documentation

### Markdown Files (.qmd)

Quarto markdown files support:
- Standard Markdown
- LaTeX math: `$inline$` or `$$display$$`
- Code blocks with syntax highlighting
- Callouts: `:::{.callout-note}` blocks
- Cross-references

Example:
```markdown
---
title: "My Page"
---

# Introduction

This is a **Quarto** document with $\LaTeX$ math: $f(x) = x^2$

::: {.callout-tip}
This is a helpful tip!
:::
```

### Jupyter Notebooks (.ipynb)

Notebooks are rendered automatically. Make sure to:
- Include a descriptive title in the first cell
- Clear unnecessary outputs before committing
- Use the `SMOKE_TEST` environment variable for fast testing

### Adding New Pages

1. Create your `.qmd` or `.ipynb` file in the appropriate directory
2. Add it to the navigation in `_quarto.yml`
3. Rebuild the documentation

## Notebook Testing

When the `SMOKE_TEST` environment variable is set, notebooks should run quickly (under 120 seconds). Use this pattern:

```python
import os

SMOKE_TEST = os.environ.get("SMOKE_TEST")
if SMOKE_TEST:
    # Fast version for testing
    n_iterations = 5
else:
    # Full version
    n_iterations = 100
```

## Troubleshooting

### Quarto not found
Ensure Quarto is installed and in your PATH:
```bash
quarto check
```

### Notebook execution errors
Check the Python environment:
```bash
python -c "import bofire; print(bofire.__version__)"
```

Disable execution for debugging:
```yaml
# In _quarto.yml
execute:
  enabled: false
```

### Build is slow
Use freeze to cache notebook outputs:
```yaml
# In _quarto.yml
execute:
  freeze: auto
```

Clear the cache if needed:
```bash
rm -rf _freeze
```

## Migration from MkDocs

This documentation was migrated from MkDocs to Quarto. The main changes:

- `.md` files → `.qmd` files
- `mkdocs.yml` → `_quarto.yml`
- `/site` output → `/_site` output
- Notebooks now directly integrated (no mkdocs-jupyter plugin needed)

## Resources

- [Quarto Documentation](https://quarto.org/docs/guide/)
- [Quarto for Python](https://quarto.org/docs/computations/python.html)
- [Quarto Websites](https://quarto.org/docs/websites/)
- [Quarto Publishing](https://quarto.org/docs/publishing/)
