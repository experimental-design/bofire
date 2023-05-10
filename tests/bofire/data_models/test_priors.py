from typing import List

import pytest
from pydantic.error_wrappers import ValidationError

from bofire.data_models.priors.api import GammaPrior, NormalPrior


def get_invalids(valid: dict) -> List[dict]:
    return [
        {k: v for k, v in valid.items() if k != k_}
        for k_ in valid.keys()
        if k_ != "type"
    ]


VALID_GAMMA_PRIOR_SPEC = {"type": "GammaPrior", "concentration": 2.0, "rate": 0.2}

VALID_NORMAL_PRIOR_SPEC = {"type": "NormalPrior", "loc": 2.0, "scale": 0.2}

PRIOR_SPECS = {
    GammaPrior: {
        "valids": [
            VALID_GAMMA_PRIOR_SPEC,
        ],
        "invalids": [
            *get_invalids(VALID_GAMMA_PRIOR_SPEC),
            *[
                {
                    **VALID_GAMMA_PRIOR_SPEC,
                    "concentration": concentration,
                    "rate": rate,
                }
                for rate in [-1.0, 0]
                for concentration in [-5, 6]
            ],
        ],
    },
    NormalPrior: {
        "valids": [
            VALID_NORMAL_PRIOR_SPEC,
        ],
        "invalids": [
            *get_invalids(VALID_NORMAL_PRIOR_SPEC),
            *[
                {
                    **VALID_NORMAL_PRIOR_SPEC,
                    "loc": 0,
                    "scale": scale,
                }
                for scale in [-1.0, 0]
            ],
        ],
    },
}


@pytest.mark.parametrize(
    "cls, spec",
    [(cls, valid) for cls, data in PRIOR_SPECS.items() for valid in data["valids"]],
)
def test_valid_prior_specs(cls, spec):
    res = cls(**spec)
    assert isinstance(res, cls)
    assert isinstance(res.__str__(), str)


@pytest.mark.parametrize(
    "cls, spec",
    [
        (cls, invalid)
        for cls, data in PRIOR_SPECS.items()
        for invalid in data["invalids"]
    ],
)
def test_invalid_prior_specs(cls, spec):
    with pytest.raises((ValueError, TypeError, KeyError, ValidationError)):
        _ = cls(**spec)
