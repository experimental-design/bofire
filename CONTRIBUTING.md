# Contributing

Contributions to BoFire are highly welcome!

## Pull Requests

Pull requests are highly welcome:

1. Create a fork from main.
2. Add or adapt unit tests according to your change.
3. Add doc-strings and update the documentation. You might consider contributing to the tutorials section.
4. Make sure that the GitHub pipelines passes.


## Development Environment

We recommend an editable installation. After cloning the repository via
```
git clone https://github.com/experimental-design/bofire.git
```
and cd `bofire`, you can proceed with
```
pip install -e ".[all]"
```
Afterwards, you can check that the tests are successful via
```
pytest tests/
```

## Coding Style
We use [Ruff](https://docs.astral.sh/ruff/) for linting, sorting and formatting of our code.
Our doc-strings are in [Google-style](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html).

In our CI/CD pipeline we check if contributions are compliant to Ruff.
To make contributors' lives easier, we have pre-commit hooks for Ruff configured in the versions corresponding to the pipeline. They can be installed via

```
pip install pre-commit
pre-commit install
```
in you local project root folder, if you want to use `pre-commit`.

## Type checks

We make heavy use of [Pydantic](https://docs.pydantic.dev/) to enforce type checks during runtime. Further, we use [Pyright](https://github.com/microsoft/pyright) for static type checking. We enforce Pyright type checks in our CI/CD pipeline.

## Tests

If you add new functionality, make sure that it is tested properly and that it does not break existing code. Our tests run in our CI/CD pipeline. The test coverage is hidden from our Readme because it is not a very robust metric. However, you can find it in the outputs of our test-CI/CD-pipeline. See [example](https://github.com/experimental-design/bofire/actions/runs/13699899620/job/38310818934#step:5:795.).

## Documentation

We use [MkDocs](https://www.mkdocs.org/) with [material theme](https://squidfunk.github.io/mkdocs-material/) and deploy our documentation to https://experimental-design.github.io/bofire/. Thereby, an API description is extracted from the doc-strings. Additionally, we have tutorials and getting-started-sections.

## License

By contributing you agree that your contributions will be licensed under the same [BSD 3-Clause License](./LICENSE) as BoFire.
