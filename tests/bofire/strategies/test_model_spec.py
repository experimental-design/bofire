import random

import pytest
from pydantic.error_wrappers import ValidationError

from bofire.strategies.strategy import KernelEnum, ModelSpec, ScalerEnum
from tests.bofire.domain.utils import get_invalids

VALID_MODEL_SPEC_SPEC = {
    "output_feature": "of1",
    "input_features": ["if1", "if2", "if3"],
    "kernel": random.choice(list(KernelEnum)),
    "ard": random.getrandbits(1),
    "scaler": random.choice(list(ScalerEnum)),
}

VALID_MODEL_SPEC_LIST = [
    ModelSpec(**VALID_MODEL_SPEC_SPEC),
    ModelSpec(**VALID_MODEL_SPEC_SPEC),
]


@pytest.mark.parametrize(
    "spec",
    [
        VALID_MODEL_SPEC_SPEC,
    ],
)
def test_valid_model_spec_specs(spec):
    ModelSpec(**spec)


@pytest.mark.parametrize(
    "spec",
    [
        *get_invalids(VALID_MODEL_SPEC_SPEC),
        {
            **VALID_MODEL_SPEC_SPEC,
            "input_features": [],
        },
        {
            **VALID_MODEL_SPEC_SPEC,
            "input_features": ["f1", "f1"],
        },
        {
            **VALID_MODEL_SPEC_SPEC,
            "input_features": ["f1", "f1", "f2"],
        },
    ],
)
def test_invalid_model_spec_specs(spec):
    with pytest.raises((ValueError, TypeError, KeyError, ValidationError)):
        ModelSpec(**spec)
