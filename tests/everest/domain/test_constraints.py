import pytest
from everest.domain.constraints import (
    ConcurrencyConstraint,
    Constraint,
    LinearConstraint,
    LinearEqualityConstraint,
    LinearInequalityConstraint,
)
from pydantic.error_wrappers import ValidationError
from tests.everest.domain.utils import INVALID_SPECS, get_invalids

VALID_CONCURRENCY_CONSTRAINT_SPEC = {
    "features": ["f1", "f2", "f3"],
    "min_count": 1,
    "max_count": 1,
    "none_also_valid": False,
}

VALID_LINEAR_CONSTRAINT_SPEC = {
    "features": ["f1", "f2", "f3"],
    "coefficients": [1, 2, 3],
    "rhs": 1.5,
}

VALID_LINEAR_CONSTRAINT_SPECS = [
    {
        **VALID_LINEAR_CONSTRAINT_SPEC,
        "features": features,
        "coefficients": coefficients,
    }
    for features, coefficients in [
        (["f1", "f2"], [-0.4, 1.4]),
    ]
]

INVALID_LINEAR_CONSTRAINT_SPECS = [
    {
        **VALID_LINEAR_CONSTRAINT_SPEC,
        "features": features,
        "coefficients": coefficients,
    }
    for features, coefficients in [
        ([], []),
        ([], [1]),
        (["f1", "f2"], [-0.4]),
        (["f1", "f2"], [-0.4, 1.4, 4.3]),
        (["f1", "f1"], [1, 1]),
        (["f1", "f1", "f2"], [1, 1, 1]),
    ]
]


CONSTRAINT_SPECS = {
    ConcurrencyConstraint: {
        "valids": [VALID_CONCURRENCY_CONSTRAINT_SPEC],
        "invalids": INVALID_SPECS + get_invalids(VALID_CONCURRENCY_CONSTRAINT_SPEC),
    },
    LinearEqualityConstraint: {
        "valids": [VALID_LINEAR_CONSTRAINT_SPEC] + VALID_LINEAR_CONSTRAINT_SPECS,
        "invalids": INVALID_SPECS
        + get_invalids(VALID_LINEAR_CONSTRAINT_SPEC)
        + INVALID_LINEAR_CONSTRAINT_SPECS,
    },
    LinearInequalityConstraint: {
        "valids": [VALID_LINEAR_CONSTRAINT_SPEC] + VALID_LINEAR_CONSTRAINT_SPECS,
        "invalids": INVALID_SPECS
        + get_invalids(VALID_LINEAR_CONSTRAINT_SPEC)
        + INVALID_LINEAR_CONSTRAINT_SPECS,
    },
}


@pytest.mark.parametrize(
    "cls, spec",
    [
        (cls, valid)
        for cls, data in CONSTRAINT_SPECS.items()
        for valid in data["valids"]
    ],
)
def test_valid_constraint_specs(cls, spec):
    res = cls(**spec)
    assert isinstance(res, cls)
    assert isinstance(res.__str__(), str)


@pytest.mark.parametrize(
    "cls, spec",
    [
        (cls, valid)
        for cls, data in CONSTRAINT_SPECS.items()
        for valid in data["valids"]
    ],
)
def test_constraint_serialize(cls, spec):
    res = cls(**spec)
    config = res.to_config()
    res2 = Constraint.from_config(config)
    assert res == res2


@pytest.mark.parametrize(
    "cls, spec",
    [
        (cls, invalid)
        for cls, data in CONSTRAINT_SPECS.items()
        for invalid in data["invalids"]
    ],
)
def test_invalid_constraint_specs(cls, spec):
    with pytest.raises((ValueError, TypeError, KeyError, ValidationError)):
        res = cls(**spec)


def test_from_greater_equal():
    c = LinearInequalityConstraint.from_greater_equal(**VALID_LINEAR_CONSTRAINT_SPEC)
    assert c.rhs == VALID_LINEAR_CONSTRAINT_SPEC["rhs"] * -1.0
    assert c.coefficients == [
        -1.0 * coef for coef in VALID_LINEAR_CONSTRAINT_SPEC["coefficients"]
    ]
    assert c.features == VALID_LINEAR_CONSTRAINT_SPEC["features"]


def test_from_smaller_equal():
    c = LinearInequalityConstraint.from_smaller_equal(**VALID_LINEAR_CONSTRAINT_SPEC)
    assert c.rhs == VALID_LINEAR_CONSTRAINT_SPEC["rhs"]
    assert c.coefficients == VALID_LINEAR_CONSTRAINT_SPEC["coefficients"]
    assert c.features == VALID_LINEAR_CONSTRAINT_SPEC["features"]
