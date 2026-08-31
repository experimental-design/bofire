# BoFire data-model migration

A batch tool that converts legacy BoFire payloads (strategy / surrogate / domain
JSON dicts) to the current schema.

## Scope

- **Offline / batch only.** No runtime adapter, no validation hooks. Callers
  read from their store, pass dicts through `migrate(...)`, and write back.
- **Pure dict -> dict.** Never calls `model_validate` internally. The output is
  a dict that the current Pydantic models accept; default-valued fields are
  left absent for Pydantic to fill in.
- **Targets: strategies and surrogates** (and bare `Domain` payloads). Other
  payload kinds are out of scope.

## Public API

```python
from bofire.data_models.migration import migrate, UnrecoverablePayloadError

migrated = migrate(
    payload,
    source="pre",          # version sentinel; see "Versioning model"
    target="0.3.3",        # default = bofire.__version__ at the time of release
    kind="strategy",       # "strategy" | "surrogate" | "domain"
)
```

`migrate` is **idempotent**: `migrate(migrate(x)) == migrate(x)`.

Errors:

- `UnrecoverablePayloadError(payload_type, reason, hint)` — raised when a
  payload references a type or value that has been removed and cannot be
  auto-migrated (e.g. `XGBoostSurrogate`, `scaler="LOG"` on the input scaler).
- `MigrationError` — base class.

## CLI

```
python -m bofire.data_models.migration \
    --kind {strategy|surrogate|domain} \
    --source pre --target 0.3.3 \
    [--input file.jsonl] [--output out.jsonl] \
    [--validate] [--dry-run] [--report report.json] \
    [--continue-on-error]
```

JSONL streaming, one record per line. Accepts both bare object lines
(`{"type": ...}`) and double-encoded JSON-string lines (`"{\"type\": ...}"`).

- `--validate` calls the current Pydantic adapter on each migrated record.
- Failures route to `<output>.failed.jsonl` (or inline with `--continue-on-error`).
- `--dry-run` writes no output, only a `--report`.

The `--report` JSON contains `total`, `ok`, `unrecoverable[]`,
`validation_failed[]`, and a `shape_histogram` of input types.

## Versioning model

The tool tracks named versions:

```
"pre"  ->  "0.3.3"  ->  (future steps)
```

`"pre"` is a sentinel meaning "any payload predating the baseline". The
legacy step is a **shape-driven defensive normalizer** because the legacy data
contains many shape variants per type (the production dump showed up to four
shapes for `SingleTaskGPSurrogate`, five for `QnehviStrategy`). There is no
per-record version tag and there will not be one — `source` is supplied by the
caller.

From `0.3.3` forward, each breaking schema PR adds a new step file
(`steps/0_3_3_to_0_3_4/`) with **strict** version-to-version transforms.

## Guiding principle for normalizers

**Mutate only what would fail validation.** Specifically:

- drop fields rejected by `extra="forbid"`,
- fix wrong-typed values (e.g. legacy `scaler="NORMALIZE"` -> `Normalize()` object),
- raise `UnrecoverablePayloadError` for removed types or unreachable states.

Never insert fields just because they have a default value — Pydantic fills
those in on validation. This keeps normalizers small, lets future schema
additions stay backwards-compatible automatically, and avoids drift between
the migration code and the data models.

## Adding a new step

1. Create `steps/<source>_to_<target>/` with one normalizer module per concern
   (e.g. `surrogates.py`, `strategies.py`).
2. Register normalizers with `@normalizer("<step_name>", "<TypeTag>", ...)`.
3. Add the step to `steps/__init__.py`'s `STEPS` list.
4. Add fixtures at `tests/bofire/data_models/migration/fixtures/<step>/...`.
5. Tests in `tests/bofire/data_models/migration/test_<step>.py` parametrize
   over the fixtures and assert `migrate(fixture) -> model_validate` succeeds.

For non-trivial schema changes, also add a row to the relevant subclass's
`RECURSE_MAP` entry in `walker.py` so the walker descends into the new field.

## Walker contract

The walker (`walker.py`) recurses **bottom-up** over typed dicts. It dispatches
on the `type` discriminator and consults `RECURSE_MAP` to find children. For
tagless containers (e.g. `BotorchSurrogates`), `STRUCTURAL_RECURSE_BY_KEY`
provides a fallback recursion spec keyed on the parent's structural key.

`INFERRED_TYPE_KEYS` lists structural keys whose child dicts have a Literal
`type` field but may have been serialized without it (e.g. `Inputs`,
`Outputs`, `Constraints`, `EngineeredFeatures`); the walker patches these in
before descending.

Recursion modes:

- `"typed"` — single typed child dict, required.
- `"typed_or_null"` — single typed child dict that may be `None`.
- `"list_of_typed"` — list of typed child dicts.
- `"container"` — single tagged container that itself has children
  (e.g. `Inputs` containing a list of features).

## Test fixtures

Fixtures live at
`tests/bofire/data_models/migration/fixtures/pre_to_0_3_3/<kind>/<type>/variant_N.json`.

They are minimally-redacted real records pulled from the production dump.
One file per `(type, field-shape)` combination observed in the wild.

Tests parametrized over fixtures verify:

1. `migrate(fixture)` produces a dict that passes `TypeAdapter.validate_python`.
2. Migration is idempotent: `migrate(migrate(x)) == migrate(x)`.
3. Known-unrecoverable types raise `UnrecoverablePayloadError`.

Walker unit tests use synthetic typed dicts.

## Deprecation policy

The `pre -> 0.3.3` normalizer is a one-shot legacy bridge. It will be removed
in BoFire **0.5.0**. By then, all downstream callers should have re-migrated
their stores and be using forward steps for any subsequent schema changes.

## Known unrecoverable payloads

- `XGBoostSurrogate` — type was removed in 0.3.x. Re-fit as
  `RandomForestSurrogate` or `LinearSurrogate`.
- `CustomSoboStrategy` — was never supported in the public API; recreate as
  a `SoboStrategy` with appropriate configuration.
- `scaler: "LOG"` or `scaler: "CHAINED_LOG_STANDARDIZE"` on the input-side
  `scaler` field — only valid on `output_scaler`. Re-fit the surrogate with
  `scaler=Normalize()` (or `None`) and put the log transform on `output_scaler`.
