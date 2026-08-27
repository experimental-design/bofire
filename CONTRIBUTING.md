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

For the data models in `bofire/data_models/` there is one additional rule: the
description of a field belongs in `Field(description=...)`, not in an `Attributes:`
block of the class doc-string. BoFire is configured entirely through these models, so
they are the API surface for users and for LLM agents alike — and only
`Field(description=...)` ends up in `model_json_schema()`, which is what an agent reads.
The class doc-string says what the model does, how it behaves and what a sensible value
looks like. Compare it with a sibling only where the two are genuinely easy to confuse
and the names do not reveal the difference. An `Examples:` block is encouraged on the
concrete types, but it is optional, is not executed by the test suite, and should set
only the fields a caller actually needs. Write it as a Google `Examples:` section with
`>>>` rather than a reStructuredText `Example::` literal block, since the doc-string is
copied verbatim into the schema description.

A field shared by every subclass is declared once, on the base class: redeclaring it in
a subclass silently drops the inherited description. For an empty container default,
prefer `Field(default=[])` over `default_factory`: pydantic deep-copies defaults, and
only the literal form appears in the JSON schema.

```python
class NChooseKConstraint(IntrapointConstraint):
    """NChooseK constraint that defines how many ingredients are allowed in a formulation.

    Bounds the *count* of active features rather than a weighted sum of their values,
    which is what distinguishes it from `LinearInequalityConstraint`.
    """

    type: Literal["NChooseKConstraint"] = "NChooseKConstraint"
    min_count: int = Field(
        description="Minimal number of non-zero/active feature values.",
    )
```

`tests/bofire/data_models/test_documentation.py` enforces this. It carries an allowlist
of the fields and classes that predate the convention, which may only shrink — the
remaining packages are being migrated one pull request at a time. If you touch a model
that is still on the allowlist, please document it and remove its entries.

In our CI/CD pipeline we check if contributions are compliant to Ruff.
To make contributors' lives easier, we have pre-commit hooks for Ruff configured in the versions corresponding to the pipeline. They can be installed via

```
pip install pre-commit
pre-commit install
```
in you local project root folder, if you want to use `pre-commit`.

## Type checks

We make heavy use of [Pydantic](https://docs.pydantic.dev/) to enforce type checks during runtime. Further, we use [ty](https://docs.astral.sh/ty/) for static type checking. We enforce ty type checks in our CI/CD pipeline.

## Validation

Data models are built from untrusted input — deserialized JSON, user code — so they validate themselves via Pydantic. Three conventions:

**Validation must work without optional extras.** `bofire.data_models` has to import and validate on the base dependencies alone; a CI job installs `pip install "." pytest` and runs `pytest tests/bofire/data_models`. A validator that reaches into rdkit, torch or entmoot fails there with an import error rather than a `ValidationError`. Call sites rarely make this visible — a method that only looks like it reports a width may generate descriptors — so prefer accessors that answer from stored fields over ones that compute.

**Ask each compatibility question once.** Where a subsystem has a natural validation entry point, let it ask all of the questions and raise `ValueError` naming the offending key. Code downstream takes the validated object and does not re-check.

**Use `assert` only for invariants, never for user errors.** `raise ValueError` for anything a user can cause; reserve `assert` for facts an earlier validation has already established. Asserts are stripped under `python -O`, so one must never be the only thing between a user and a wrong answer.

## Tests

If you add new functionality, make sure that it is tested properly and that it does not break existing code. Our tests run in our CI/CD pipeline. The test coverage is hidden from our Readme because it is not a very robust metric. However, you can find it in the outputs of our test-CI/CD-pipeline. See [example](https://github.com/experimental-design/bofire/actions/runs/13699899620/job/38310818934#step:5:795.).

## Documentation

We use [Quarto](https://quarto.org/) and [Quartodoc](https://github.com/machow/quartodoc) deploy our documentation to https://experimental-design.github.io/bofire/. Thereby, an API description is extracted from the doc-strings. Additionally, we have tutorials and getting-started-sections. To build the documentation locally, install Quarto and Quartodoc and run

```
quartodoc build
quarto render
```

Optionally, you can use the environment variable `SMOKE_TEST=1` if you just want a test build. Otherwise, some benchmarks will take some time until they finish.

## License

By contributing you agree that your contributions will be licensed under the same [BSD 3-Clause License](./LICENSE) as BoFire.
