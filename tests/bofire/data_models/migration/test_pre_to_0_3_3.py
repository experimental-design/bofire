import json
from pathlib import Path

import pytest
from pydantic import TypeAdapter

from bofire.data_models.domain.api import Domain
from bofire.data_models.migration import UnrecoverablePayloadError, migrate
from bofire.data_models.strategies.api import AnyStrategy
from bofire.data_models.surrogates.api import AnySurrogate


FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "pre_to_0_3_3"

# Payload types known to be unrecoverable. These fixtures are exercised by a
# separate test that asserts UnrecoverablePayloadError is raised.
UNRECOVERABLE = {
    ("surrogate", "XGBoostSurrogate"),
    ("strategy", "CustomSoboStrategy"),
}

_ADAPTERS = {
    "surrogate": TypeAdapter(AnySurrogate),
    "strategy": TypeAdapter(AnyStrategy),
    "domain": TypeAdapter(Domain),
}


def _collect_fixtures():
    items = []
    for kind_dir in sorted(FIXTURE_ROOT.iterdir()):
        if not kind_dir.is_dir():
            continue
        kind = kind_dir.name
        for type_dir in sorted(kind_dir.iterdir()):
            if not type_dir.is_dir():
                continue
            type_name = type_dir.name
            for variant_file in sorted(type_dir.glob("variant_*.json")):
                items.append((kind, type_name, variant_file))
    return items


_ALL = _collect_fixtures()
_VALID = [(k, t, p) for (k, t, p) in _ALL if (k, t) not in UNRECOVERABLE]
_UNRECOVERABLE = [(k, t, p) for (k, t, p) in _ALL if (k, t) in UNRECOVERABLE]


@pytest.mark.parametrize(
    "kind,type_name,variant_path",
    _VALID,
    ids=[f"{k}-{t}-{p.stem}" for k, t, p in _VALID],
)
def test_fixture_migrates_and_validates(kind, type_name, variant_path):
    payload = json.loads(variant_path.read_text())
    migrated = migrate(payload, source="pre", target="0.3.3", kind=kind)
    _ADAPTERS[kind].validate_python(migrated)


@pytest.mark.parametrize(
    "kind,type_name,variant_path",
    _VALID,
    ids=[f"{k}-{t}-{p.stem}" for k, t, p in _VALID],
)
def test_migration_is_idempotent(kind, type_name, variant_path):
    payload = json.loads(variant_path.read_text())
    once = migrate(payload, source="pre", target="0.3.3", kind=kind)
    twice = migrate(
        json.loads(json.dumps(once)), source="pre", target="0.3.3", kind=kind
    )
    assert once == twice


@pytest.mark.parametrize(
    "kind,type_name,variant_path",
    _UNRECOVERABLE,
    ids=[f"{k}-{t}-{p.stem}" for k, t, p in _UNRECOVERABLE],
)
def test_unrecoverable_fixtures_raise(kind, type_name, variant_path):
    payload = json.loads(variant_path.read_text())
    with pytest.raises(UnrecoverablePayloadError):
        migrate(payload, source="pre", target="0.3.3", kind=kind)


def test_scaler_log_on_input_scaler_unrecoverable():
    payload = {
        "type": "SingleTaskGPSurrogate",
        "inputs": {
            "type": "Inputs",
            "features": [{"type": "ContinuousInput", "key": "x", "bounds": [0.0, 1.0]}],
        },
        "outputs": {
            "type": "Outputs",
            "features": [{"type": "ContinuousOutput", "key": "y", "objective": None}],
        },
        "scaler": "LOG",  # only valid on output_scaler, not scaler
        "kernel": {"type": "MaternKernel", "ard": True, "nu": 2.5},
        "noise_prior": {"type": "GammaPrior", "concentration": 1.1, "rate": 0.05},
    }
    with pytest.raises(UnrecoverablePayloadError):
        migrate(payload, kind="surrogate")
