# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BoFire (**B**ayesian **O**ptimization **F**ramework **I**ntended for **R**eal **E**xperiments) is a Python framework for experimental design, combining Design of Experiments (DoE) and Bayesian Optimization (BO). It supports mixed continuous/discrete/categorical parameter spaces, single and multi-objective optimization, and chemical encodings for molecular optimization.

## Build & Test Commands

```bash
# Install for development (full)
pip install -e ".[all]"

# Install core optimization only
pip install -e ".[optimization]"

# Run all tests
pytest tests/

# Run data model tests only (faster, no torch required)
pytest tests/bofire/data_models

# Run with coverage
pytest -ra --cov=bofire --cov-report term-missing tests

# Run a single test file
pytest tests/bofire/strategies/test_sobo.py

# Run a single test
pytest tests/bofire/strategies/test_sobo.py::test_function_name
```

## Linting & Type Checking

```bash
# Install pre-commit hooks (recommended)
pip install pre-commit
pre-commit install

# Run all linting/formatting
pre-commit run --all-files

# Or run ruff directly
ruff check .
ruff format .

# Type checking — use the version CI pins, not whatever `uvx ty` resolves to.
# Look up the pin in .github/workflows/lint.yaml (the `ty==` install), then:
uvx --from ty==<pinned-version> ty check bofire
```

ty is pre-1.0 and its rules tighten fast, so an unpinned run reports a substantially
different set of diagnostics than CI and will not tell you whether Lint passes. Rule
configuration lives in `pyproject.toml` under `[tool.ty.rules]` / `[tool.ty.analysis]`.

## Architecture

### Data Models vs Functional Separation

BoFire separates serializable data models (Pydantic) from functional implementations. This enables REST API integration.

- `bofire/data_models/` - Pydantic models for serialization
- `bofire/strategies/`, `bofire/surrogates/`, etc. - Functional implementations

### Key Modules

- **data_models/** - Pydantic schemas for all BoFire objects
  - `features/` - Input/output feature definitions (ContinuousInput, CategoricalOutput, etc.)
  - `domain/` - Domain composed of Inputs + Outputs + Constraints
  - `objectives/` - MinimizeObjective, MaximizeObjective, CloseToTargetObjective
  - `constraints/` - Linear, nonlinear, and black-box constraints
  - `surrogates/` - Surrogate model specifications
  - `strategies/` - Strategy configurations
  - `acquisition_functions/` - BoTorch acquisition functions (qLogEI, etc.)

- **strategies/** - Optimization strategy implementations
  - Uses ask/tell interface: `strategy.ask(n)` returns candidates, `strategy.tell(experiments)` updates model
  - `botorch/` - BoTorch-based strategies (SoboStrategy, MoboStrategy)
  - `doe/` - Design of Experiments strategies
  - `samplers/` - Sampling methods for constrained spaces

- **surrogates/** - Surrogate model implementations
  - `single_task_gp.py`, `multi_task_gp.py` - Gaussian Processes
  - `random_forest.py`, `mlp.py` - Alternative surrogates

- **kernels/** - GP kernel implementations including molecular kernels

### API Pattern

Each module typically has an `api.py` file that re-exports public interfaces:
```python
from bofire.data_models.features.api import ContinuousInput, CategoricalOutput
from bofire.strategies.api import SoboStrategy
from bofire.data_models.acquisition_functions.api import qLogEI
```

## Test Style

**Prefer plain test functions over test classes.** Roughly 85% of BoFire's tests are
module-level `def test_*` functions (~815 functions vs ~150 methods in classes); only
four files use classes at all (`tests/bofire/test_register.py`,
`tests/bofire/strategies/test_utils.py`, `tests/bofire/strategies/test_nchoosek_pruning.py`,
`tests/bofire/utils/test_domain_repair.py`).

Use a function unless you need a shared `pytest.fixture` for setup that several tests
in the group depend on. When adding to a file that is already consistently class-based,
follow that file's convention instead — consistency within a file beats the global default.

Prefer `@pytest.mark.parametrize` over near-duplicate test bodies.

**Put tests in the module that matches the code under test.** Tests for feature
containers belong in `tests/bofire/data_models/domain/test_features.py`, not in
`tests/bofire/test_register.py` — the latter is for the registration machinery itself.

## Data Model Testing

Data models use a spec-based parametrized testing system. The infrastructure lives in `tests/bofire/data_models/specs/`.

### Spec System

The core classes in `tests/bofire/data_models/specs/specs.py`:

- **`Spec(cls, spec_callable)`** — wraps a data model class and a lambda that returns a valid spec dict
- **`InvalidSpec(cls, spec_callable, error, message)`** — wraps an invalid spec with expected error
- **`Specs(invalidators)`** — collection that holds valid/invalid specs; use `add_valid()` and `add_invalid()`

### Serialization Roundtrip Contract

The key invariant enforced by `tests/bofire/data_models/serialization/test_serialization.py`:

```python
spec = some_spec.typed_spec()       # spec dict + {"type": ClassName}
obj = SomeClass(**spec)             # instantiate from spec
assert obj.model_dump() == spec     # EXACT match required
```

**This means:** every field that appears in `model_dump()` output must be present in the spec dict. When adding a new field with a default value to a base class (e.g., `context: Optional[str] = None` on `Feature`), you must add that field with its default to **every `add_valid()` spec** for all subclasses.

### Spec File Patterns

**Leaf specs** (single objects) — spec dicts contain plain values:
```python
# tests/bofire/data_models/specs/features.py
specs.add_valid(
    features.ContinuousInput,
    lambda: {
        "key": str(uuid.uuid4()),
        "bounds": [3, 5.3],
        "unit": None,
        "context": None,   # all fields with defaults must be explicit
    },
)
```

**Container specs** (nested objects) — use `.model_dump()` for children:
```python
# tests/bofire/data_models/specs/domain.py
specs.add_valid(
    Domain,
    lambda: {
        "inputs": Inputs(features=[...]).model_dump(),    # nested via model_dump()
        "outputs": Outputs(features=[...]).model_dump(),
        "constraints": Constraints().model_dump(),
        "context": None,
    },
)
```

Container specs (inputs.py, outputs.py, constraints_container.py, engineered_features.py) don't need manual updates when a field is added to a leaf class, because `.model_dump()` on the nested objects already includes all fields.

### Invalid Specs

Invalid specs test that construction raises the expected error. They do **not** test serialization, so they don't need every default field:

```python
specs.add_invalid(
    features.ContinuousInput,
    lambda: {"key": "a", "bounds": [5, 3]},  # no need for "context" etc.
    error=ValueError,
    message="Sequence is not monotonically increasing.",
)
```

### Deserialization Tests

`tests/bofire/data_models/serialization/test_deserialization.py` tests round-trip through `TypeAdapter`:
```python
obj = spec.obj()
deserialized = TypeAdapter(AnyFeature).validate_python(obj.model_dump())
assert obj == deserialized
```

### Fixtures

Specs are wired to pytest via `tests/bofire/conftest.py`:
```python
@fixture(params=specs.features.valids)
def feature_spec(request) -> Spec:
    return request.param
```

### Documentation Ratchet

`tests/bofire/data_models/test_documentation.py` requires every field declared on a data
model class to carry `Field(description=...)`, and every class to carry a docstring. It
enumerates classes by walking `bofire.data_models` with `pkgutil`, so a new package is
covered the moment it exists.

Two things about how it is written matter if you touch it:

- It checks **own** fields (those in `cls.__annotations__`), not all of
  `cls.model_fields`. Inherited fields are documented on the class that declares them;
  checking them again would report one undocumented base field once per subclass.
- The two `UNDOCUMENTED_*` sets record what predates the convention and **may only
  shrink**. `test_field_allowlist_has_no_stale_entries` fails once an entry is
  documented but not deleted, so documenting a package and pruning its entries happen in
  the same commit. Never add to these sets.

The migration is running package by package; `constraints` is done.

### Checklist: Adding a Field to a Base Data Model Class

1. Add the field to the class (e.g., `Feature`, `Constraint`, `Domain`) with
   `Field(description=...)`
2. Add `"field_name": default_value` to every `add_valid()` spec for that class and all subclasses
3. Invalid specs and container specs using `.model_dump()` don't need changes
4. Run `pytest tests/bofire/data_models` to verify

### Checklist: Adding a New Data Model Class

BoFire uses a two-layer architecture: data models (Pydantic) and functional implementations, connected by type unions, registration functions, and mappers.

#### 1. Create the data model class

Create a new file or add to an existing one in `bofire/data_models/{domain}/`. Every data model needs a `type` literal discriminator:

```python
# bofire/data_models/kernels/my_kernel.py
from typing import Literal
from pydantic import Field
from bofire.data_models.kernels.kernel import ContinuousKernel

class MyCustomKernel(ContinuousKernel):
    """One line on what this kernel is, and when to pick it over its siblings."""

    type: Literal["MyCustomKernel"] = "MyCustomKernel"
    my_param: float = Field(description="What this parameter controls.")
```

#### 2. Register in the type union

Each domain has an `api.py` that defines type unions (e.g., `AnyFeature`, `AnyConstraint`, `AnyKernel`). These unions use `Union[tuple(type_list)]` where the list is mutable.

**Option A — Static registration:** Add the import and class to the union list in `bofire/data_models/{domain}/api.py`.

**Option B — Dynamic registration:** Use the `register_*()` functions in `bofire/data_models/{domain}/_register.py`. These handle cascading Pydantic model rebuilds automatically:

```python
from bofire.data_models.kernels._register import register_kernel
register_kernel(MyCustomKernel)
```

Available registration functions:
- `bofire/data_models/features/_register.py` → `register_engineered_feature()`
- `bofire/data_models/strategies/_register.py` → `register_strategy()`
- `bofire/data_models/kernels/_register.py` → `register_kernel()`
- `bofire/data_models/priors/_register.py` → `register_prior()`, `register_prior_constraint()`
- `bofire/data_models/surrogates/botorch_surrogates.py` → `register_botorch_surrogate()`

These functions are idempotent (check if already registered) and cascade rebuilds to dependent models (e.g., registering a kernel rebuilds surrogate models that reference kernels).

#### 3. Create the functional implementation and register the mapper

Each domain has a `mapper.py` that maps data model classes to implementations:

```python
# bofire/strategies/mapper.py or bofire/surrogates/mapper.py
from bofire.strategies.mapper import register

@register(data_model_cls=MyStrategyDataModel)
class MyStrategy(Strategy):
    ...
```

Or function-based (kernels, priors):
```python
# bofire/kernels/mapper.py
KERNEL_MAP[MyCustomKernel] = map_my_custom_kernel
```

The `register()` decorators in `bofire/strategies/api.py` and `bofire/surrogates/api.py` handle both the mapper registration and the data model registration in one step.

#### 4. Add test specs

Add entries to the appropriate file in `tests/bofire/data_models/specs/`:

```python
# tests/bofire/data_models/specs/kernels.py
specs.add_valid(
    MyCustomKernel,
    lambda: {"my_param": 1.0},
)
```

The spec is automatically picked up by `tests/bofire/conftest.py` which imports all specs from `tests/bofire/data_models/specs/api.py` and parametrizes fixtures over `specs.{domain}.valids` / `specs.{domain}.invalids`. No conftest changes needed unless adding an entirely new domain.

#### 5. Key files reference

| Domain | Data Model | Type Union / Registration | Mapper | Test Specs |
|--------|-----------|--------------------------|--------|------------|
| Features | `bofire/data_models/features/` | `api.py`, `_register.py` | N/A | `tests/.../specs/features.py` |
| Constraints | `bofire/data_models/constraints/` | `api.py` | N/A | `tests/.../specs/constraints.py` |
| Strategies | `bofire/data_models/strategies/` | `api.py`, `_register.py` | `bofire/strategies/mapper.py` | `tests/.../specs/strategies.py` |
| Surrogates | `bofire/data_models/surrogates/` | `api.py`, `botorch_surrogates.py` | `bofire/surrogates/mapper.py` | `tests/.../specs/surrogates.py` |
| Kernels | `bofire/data_models/kernels/` | `api.py`, `_register.py` | `bofire/kernels/mapper.py` | `tests/.../specs/kernels.py` |
| Priors | `bofire/data_models/priors/` | `api.py`, `_register.py` | `bofire/priors/mapper.py` | `tests/.../specs/priors.py` |

## Code Style

- **Linter/Formatter**: Ruff (line length 88)
- **Docstrings**: Google-style
- **Type Checking**: Pydantic for runtime, ty for static analysis
- **Python**: 3.10+
- **Pydantic validators**: public methods named `validate_*`, no leading underscore
  (111 validators in `bofire/`, 104 use the `validate_` prefix). Module-level helper
  functions they call may still be private. No linter enforces this.
- **Don't restate the annotation in a validator.** A `field_validator` defaults to
  `mode="after"`, so pydantic has already validated *and coerced* the value: re-checking
  or re-coercing it there is unreachable code, and its error message can never surface.
  Validate only what the type cannot express — cross-field consistency, or domain
  semantics such as "is this string a parseable SMILES". Use `mode="before"` when you
  genuinely need the raw input.
- **Docstrings** state what the code does now: arguments, returns, raises, invariants
  callers may rely on. Design history, migration rationale ("rather than stored",
  "this is why X asserts") and enumerated call sites belong in the commit message and
  CHANGELOG — in a docstring they go stale and crowd out the contract.
- **Data model field prose goes in `Field(description=...)`, never an `Attributes:`
  block.** Only the former reaches `model_json_schema()`, which is the API surface
  agents read. The class docstring says what the model is and when to pick it over its
  siblings in the union; an `Examples:` block is encouraged on concrete types but is
  optional and not executed. The `type` discriminator needs no description.
  `tests/bofire/data_models/test_documentation.py` enforces this — see
  [Documentation Ratchet](#documentation-ratchet).

## Documentation

```bash
# Build API docs
quartodoc build

# Render full docs
quarto render

# Preview with live reload
quarto preview

# Fast smoke test build
SMOKE_TEST=1 quarto render
```
