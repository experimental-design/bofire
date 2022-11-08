import pytest
from pydantic.error_wrappers import ValidationError

from bofire.domain.desirability_functions import (
    CloseToTargetDesirabilityFunction,
    ConstantDesirabilityFunction,
    DeltaIdentityDesirabilityFunction,
    DesirabilityFunction,
    MaxIdentityDesirabilityFunction,
    MaxSigmoidDesirabilityFunction,
    MinIdentityDesirabilityFunction,
    MinSigmoidDesirabilityFunction,
    TargetDesirabilityFunction,
)
from tests.bofire.domain.utils import INVALID_SPECS, get_invalids

VALID_CONSTANT_DESIRABILITY_FUNCTION_SPEC = {
    "w": 100.0,
}

INVALID_CONSTANT_DESIRABILITY_FUNCTION_SPEC = {
    "w": "s",
}

VALID_IDENTITY_DESIRABILITY_FUNCTION_SPEC = {"w": 0.5}

VALID_DELTA_IDENTITY_DESIRABILITY_FUNCTION_SPEC = {"w": 0.5, "ref_point": 5}

VALID_SIGMOID_DESIRABILITY_FUNCTION_SPEC = {"steepness": 5, "tp": -44.1, "w": 0.5}

VALID_TARGET_DESIRABILITY_FUNCTION_SPEC = {
    "target_value": -42,
    "steepness": 5,
    "tolerance": 100,
    "w": 0.5,
}

VALID_CLOSE_TO_TARGET_DESIRABILITY_FUNCTION_SPEC = {
    "target_value": 42,
    "exponent": 2,
    "tolerance": 1.5,
    "w": 1.0,
}

INVALID_W = [
    {"w": 0},
    {"w": -100},
    {"w": 1.0000001},
    {"w": 100},
]

INVALID_BOUNDS = [
    {"lower_bound": 5, "upper_bound": 3},
    {"lower_bound": None, "upper_bound": None},
]
INVALID_STEEPNESS = [
    {"steepness": 0},
    {"steepness": -100},
]

INVALID_TOLERANCE = [
    {"tolerance": -0.1},
    {"tolerance": -100},
]

DESIRABILITY_FUNCTION_SPECS = {
    MinIdentityDesirabilityFunction: {
        "valids": [VALID_IDENTITY_DESIRABILITY_FUNCTION_SPEC],
        "invalids": INVALID_SPECS
        + get_invalids(VALID_IDENTITY_DESIRABILITY_FUNCTION_SPEC)
        + [
            {
                **VALID_IDENTITY_DESIRABILITY_FUNCTION_SPEC,
                **invalid,
            }
            for invalid in [
                *INVALID_W,
                *INVALID_BOUNDS,
            ]
        ],
    },
    MaxIdentityDesirabilityFunction: {
        "valids": [VALID_IDENTITY_DESIRABILITY_FUNCTION_SPEC],
        "invalids": INVALID_SPECS
        + get_invalids(VALID_IDENTITY_DESIRABILITY_FUNCTION_SPEC)
        + [
            {
                **VALID_IDENTITY_DESIRABILITY_FUNCTION_SPEC,
                **invalid,
            }
            for invalid in [*INVALID_W, *INVALID_BOUNDS]
        ],
    },
    DeltaIdentityDesirabilityFunction: {
        "valids": [VALID_DELTA_IDENTITY_DESIRABILITY_FUNCTION_SPEC],
        "invalids": INVALID_SPECS
        + get_invalids(VALID_DELTA_IDENTITY_DESIRABILITY_FUNCTION_SPEC)
        + [
            {
                **VALID_DELTA_IDENTITY_DESIRABILITY_FUNCTION_SPEC,
                **invalid,
            }
            for invalid in [*INVALID_W, *INVALID_BOUNDS]
        ],
    },
    MinSigmoidDesirabilityFunction: {
        "valids": [VALID_SIGMOID_DESIRABILITY_FUNCTION_SPEC],
        "invalids": INVALID_SPECS
        + get_invalids(VALID_SIGMOID_DESIRABILITY_FUNCTION_SPEC)
        + [
            {**VALID_SIGMOID_DESIRABILITY_FUNCTION_SPEC, **invalid}
            for invalid in [
                *INVALID_W,
                *INVALID_STEEPNESS,
            ]
        ],
    },
    MaxSigmoidDesirabilityFunction: {
        "valids": [VALID_SIGMOID_DESIRABILITY_FUNCTION_SPEC],
        "invalids": INVALID_SPECS
        + get_invalids(VALID_SIGMOID_DESIRABILITY_FUNCTION_SPEC)
        + [
            {**VALID_SIGMOID_DESIRABILITY_FUNCTION_SPEC, **invalid}
            for invalid in [
                *INVALID_W,
                *INVALID_STEEPNESS,
            ]
        ],
    },
    TargetDesirabilityFunction: {
        "valids": [VALID_TARGET_DESIRABILITY_FUNCTION_SPEC],
        "invalids": INVALID_SPECS
        + get_invalids(VALID_TARGET_DESIRABILITY_FUNCTION_SPEC)
        + [
            {**VALID_TARGET_DESIRABILITY_FUNCTION_SPEC, **invalid}
            for invalid in [
                *INVALID_STEEPNESS,
                *INVALID_TOLERANCE,
                *INVALID_W,
            ]
        ],
    },
    CloseToTargetDesirabilityFunction: {
        "valids": [VALID_CLOSE_TO_TARGET_DESIRABILITY_FUNCTION_SPEC],
        "invalids": INVALID_SPECS
        + get_invalids(VALID_CLOSE_TO_TARGET_DESIRABILITY_FUNCTION_SPEC)
        + [
            {**VALID_CLOSE_TO_TARGET_DESIRABILITY_FUNCTION_SPEC, **invalid}
            for invalid in [
                # *INVALID_STEEPNESS,
                *INVALID_TOLERANCE,
                *INVALID_W,
            ]
        ],
    },
    ConstantDesirabilityFunction: {
        "valids": [VALID_CONSTANT_DESIRABILITY_FUNCTION_SPEC],
        "invalids": [INVALID_CONSTANT_DESIRABILITY_FUNCTION_SPEC],
    },
}


@pytest.mark.parametrize(
    "cls, spec",
    [
        (cls, valid)
        for cls, data in DESIRABILITY_FUNCTION_SPECS.items()
        for valid in data["valids"]
    ],
)
def test_valid_desirability_function_specs(cls, spec):
    res = cls(**spec)
    assert isinstance(res, cls)


@pytest.mark.parametrize(
    "cls, spec",
    [
        (cls, valid)
        for cls, data in DESIRABILITY_FUNCTION_SPECS.items()
        for valid in data["valids"]
    ],
)
def test_desirability_function_serialize(cls, spec):
    res = cls(**spec)
    config = res.to_config()
    res2 = DesirabilityFunction.from_config(config)
    assert res == res2


@pytest.mark.parametrize(
    "cls, spec",
    [
        (cls, invalid)
        for cls, data in DESIRABILITY_FUNCTION_SPECS.items()
        for invalid in data["invalids"]
    ],
)
def test_invalid_desirability_function_specs(cls, spec):
    with pytest.raises((ValueError, TypeError, KeyError, ValidationError)):
        _ = cls(**spec)
