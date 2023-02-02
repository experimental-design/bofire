# Contributing

Contributions to BoFire are highly welcome!

## Development Enviromnent

We recommend an editable installation. After cloning the repository via
```
git clone https://github.com/experimental-design/bofire.git
```
and cd `bofire`, you can proceed with
```
pip install -e .[testing]
```
Afterwards, you can check that the tests are successful via
```
pytest tests/
```
## Coding Style
We format our code with [Black](https://github.com/psf/black).
```
pip install black
``` 
Our doc-strings are in [Google-style](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html).
Further, we use [Flake8](https://flake8.pycqa.org/en/latest/) for coding-style enforcement.
```
pip install flake8
```
Imports are sorted via [Isort](https://github.com/PyCQA/isort).
```
pip install isort
```

In our CI/CD pipeline we check if contributions are compliant to Black, Flake8, and Isort. To make contributors' lives easier,
we have pre-commit hooks for Black, Flake8, and Isort configured in the versions corresponding to the pipeline. Pre-commit hooks can be installed via

```
pip install pre-commit
pre-commit install
```

## Type checks

We make heavily use of [Pydantic](https://docs.pydantic.dev/) to enforce type checks during runtime. Further, we use [Pyright](https://github.com/microsoft/pyright) for static type checking. We enforce Pyright type checks in our CI/CD pipeline.

## Documentation

We use [MkDocs](https://www.mkdocs.org/) with [material theme](https://squidfunk.github.io/mkdocs-material/) and deploy our documentation to https://experimental-design.github.io/bofire/. Thereby, an API description is extracted from the doc-strings. Additionally, we have tutorials and getting-started-sections.
