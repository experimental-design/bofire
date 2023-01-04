import pytest
from pydantic.error_wrappers import ValidationError

from bofire.domain.constraints import (
    Constraint,
    Constraints,
    LinearConstraint,
    LinearEqualityConstraint,
    LinearInequalityConstraint,
    NChooseKConstraint,
    NonlinearEqualityConstraint,
    NonlinearInequalityConstraint,
)
from bofire.domain.features import ContinuousInput, ContinuousOutput, InputFeatures
from bofire.utils.enum import SamplingMethodEnum
from tests.bofire.domain.utils import INVALID_SPECS, get_invalids

VALID_NCHOOSEKE_CONSTRAINT_SPEC = {
    "features": ["f1", "f2", "f3"],
    "min_count": 1,
    "max_count": 1,
    "none_also_valid": False,
}

VALID_LINEAR_EQUALITY_CONSTRAINT_SPEC = {
    "type": "LinearEqualityConstraint",
    "features": ["f1", "f2", "f3"],
    "coefficients": [1, 2, 3],
    "rhs": 1.5,
}

VALID_LINEAR_INEQUALITY_CONSTRAINT_SPEC = {
    "type": "LinearInequalityConstraint",
    "features": ["f1", "f2", "f3"],
    "coefficients": [1, 2, 3],
    "rhs": 1.5,
}

VALID_NONLINEAR_EQUALITY_CONSTRAINT_SPEC = {
    "type": "NonlinearEqualityConstraint",
    "features": ["f1", "f2"],
    "expression": "f1*f2",
}

VALID_NONLINEAR_INEQUALITY_CONSTRAINT_SPEC = {
    "type": "NonlinearInequalityConstraint",
    "features": ["f1", "f2"],
    "expression": "f1*f2",
}

INVALID_NONLINEAR_EQUALITY_CONSTRAINT_SPEC = {
    "type": "NonlinearEqualityConstraint",
    "expression": [5, 7, 8],
}

INVALID_NONLINEAR_INEQUALITY_CONSTRAINT_SPEC = {
    "type": "NonlinearInequalityConstraint",
    "expression": [5, 7, 8],
}

VALID_LINEAR_EQUALITY_CONSTRAINT_SPECS = [
    {
        **VALID_LINEAR_EQUALITY_CONSTRAINT_SPEC,
        "features": features,
        "coefficients": coefficients,
    }
    for features, coefficients in [
        (["f1", "f2"], [-0.4, 1.4]),
    ]
]

VALID_LINEAR_INEQUALITY_CONSTRAINT_SPECS = [
    {
        **VALID_LINEAR_INEQUALITY_CONSTRAINT_SPEC,
        "features": features,
        "coefficients": coefficients,
    }
    for features, coefficients in [
        (["f1", "f2"], [-0.4, 1.4]),
    ]
]

INVALID_LINEAR_EQUALITY_CONSTRAINT_SPECS = [
    {
        **VALID_LINEAR_EQUALITY_CONSTRAINT_SPEC,
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

INVALID_LINEAR_INEQUALITY_CONSTRAINT_SPECS = [
    {
        **VALID_LINEAR_INEQUALITY_CONSTRAINT_SPEC,
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
    NChooseKConstraint: {
        "valids": [VALID_NCHOOSEKE_CONSTRAINT_SPEC],
        "invalids": INVALID_SPECS + get_invalids(VALID_NCHOOSEKE_CONSTRAINT_SPEC),
    },
    LinearEqualityConstraint: {
        "valids": [VALID_LINEAR_EQUALITY_CONSTRAINT_SPEC]
        + VALID_LINEAR_EQUALITY_CONSTRAINT_SPECS,
        "invalids": INVALID_SPECS
        + get_invalids(VALID_LINEAR_EQUALITY_CONSTRAINT_SPEC)
        + INVALID_LINEAR_EQUALITY_CONSTRAINT_SPECS,
    },
    LinearInequalityConstraint: {
        "valids": [VALID_LINEAR_INEQUALITY_CONSTRAINT_SPEC]
        + VALID_LINEAR_INEQUALITY_CONSTRAINT_SPECS,
        "invalids": INVALID_SPECS
        + get_invalids(VALID_LINEAR_INEQUALITY_CONSTRAINT_SPEC)
        + INVALID_LINEAR_INEQUALITY_CONSTRAINT_SPECS,
    },
    NonlinearEqualityConstraint: {
        "valids": [VALID_NONLINEAR_EQUALITY_CONSTRAINT_SPEC],
        "invalids": [INVALID_NONLINEAR_EQUALITY_CONSTRAINT_SPEC],
    },
    NonlinearInequalityConstraint: {
        "valids": [VALID_NONLINEAR_INEQUALITY_CONSTRAINT_SPEC],
        "invalids": [INVALID_NONLINEAR_INEQUALITY_CONSTRAINT_SPEC],
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
    res2 = Constraint.from_dict(spec)
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
        _ = cls(**spec)


def test_from_greater_equal():
    c = LinearInequalityConstraint.from_greater_equal(
        features=VALID_LINEAR_INEQUALITY_CONSTRAINT_SPEC["features"],
        coefficients=VALID_LINEAR_INEQUALITY_CONSTRAINT_SPEC["coefficients"],
        rhs=VALID_LINEAR_INEQUALITY_CONSTRAINT_SPEC["rhs"],
    )
    assert c.rhs == VALID_LINEAR_INEQUALITY_CONSTRAINT_SPEC["rhs"] * -1.0
    assert c.coefficients == [
        -1.0 * coef for coef in VALID_LINEAR_INEQUALITY_CONSTRAINT_SPEC["coefficients"]
    ]
    assert c.features == VALID_LINEAR_INEQUALITY_CONSTRAINT_SPEC["features"]


def test_as_greater_equal():
    c = LinearInequalityConstraint.from_greater_equal(
        features=VALID_LINEAR_INEQUALITY_CONSTRAINT_SPEC["features"],
        coefficients=VALID_LINEAR_INEQUALITY_CONSTRAINT_SPEC["coefficients"],
        rhs=VALID_LINEAR_INEQUALITY_CONSTRAINT_SPEC["rhs"],
    )
    features, coefficients, rhs = c.as_greater_equal()
    assert c.rhs == rhs * -1.0
    assert coefficients == [-1.0 * coef for coef in c.coefficients]
    assert c.features == features


def test_from_smaller_equal():
    c = LinearInequalityConstraint.from_smaller_equal(
        features=VALID_LINEAR_INEQUALITY_CONSTRAINT_SPEC["features"],
        coefficients=VALID_LINEAR_INEQUALITY_CONSTRAINT_SPEC["coefficients"],
        rhs=VALID_LINEAR_INEQUALITY_CONSTRAINT_SPEC["rhs"],
    )
    assert c.rhs == VALID_LINEAR_INEQUALITY_CONSTRAINT_SPEC["rhs"]
    assert c.coefficients == VALID_LINEAR_INEQUALITY_CONSTRAINT_SPEC["coefficients"]
    assert c.features == VALID_LINEAR_INEQUALITY_CONSTRAINT_SPEC["features"]


def test_as_smaller_equal():
    c = LinearInequalityConstraint.from_smaller_equal(
        features=VALID_LINEAR_INEQUALITY_CONSTRAINT_SPEC["features"],
        coefficients=VALID_LINEAR_INEQUALITY_CONSTRAINT_SPEC["coefficients"],
        rhs=VALID_LINEAR_INEQUALITY_CONSTRAINT_SPEC["rhs"],
    )
    features, coefficients, rhs = c.as_smaller_equal()
    assert c.rhs == rhs
    assert coefficients == c.coefficients
    assert c.features == features


# test the Constraints Class
c1 = LinearEqualityConstraint(**VALID_LINEAR_EQUALITY_CONSTRAINT_SPEC)
c2 = LinearInequalityConstraint(**VALID_LINEAR_INEQUALITY_CONSTRAINT_SPEC)
c3 = NChooseKConstraint(**VALID_NCHOOSEKE_CONSTRAINT_SPEC)
c4 = NonlinearEqualityConstraint(**VALID_NONLINEAR_EQUALITY_CONSTRAINT_SPEC)
c5 = NonlinearInequalityConstraint(**VALID_NONLINEAR_INEQUALITY_CONSTRAINT_SPEC)
c6 = LinearInequalityConstraint.from_smaller_equal(
    features=["f1", "f2", "f3"], coefficients=[1, 1, 1], rhs=100.0
)

if1 = ContinuousInput(key="f1", lower_bound=0, upper_bound=2)
if2 = ContinuousInput(key="f2", lower_bound=0, upper_bound=4)
if3 = ContinuousInput(key="f3", lower_bound=3, upper_bound=8)

input_features = InputFeatures(features=[if1, if2, if3])

constraints = Constraints(constraints=[c1, c2, c3])
constraints2 = Constraints(constraints=[c4, c5])
constraints3 = Constraints(constraints=[c6])


@pytest.mark.parametrize("constraints", [constraints, constraints2, constraints3])
def test_constraints_serialize(constraints):
    nconstraints = Constraints(**constraints.dict())
    assert constraints == nconstraints


@pytest.mark.parametrize(
    "constraints",
    [
        (["s"]),
        ([LinearInequalityConstraint(**VALID_LINEAR_INEQUALITY_CONSTRAINT_SPEC)], 5),
        (
            [LinearInequalityConstraint(**VALID_LINEAR_INEQUALITY_CONSTRAINT_SPEC)],
            ContinuousOutput(key="s"),
        ),
    ],
)
def test_constraints_invalid_constraint(constraints):
    with pytest.raises((ValueError, TypeError, KeyError, ValidationError)):
        Constraints(constraints=constraints)


@pytest.mark.parametrize(
    "constraints, ConstraintType, exact, expected",
    [
        (constraints, LinearConstraint, True, []),
        (constraints, LinearConstraint, False, [c1, c2]),
        (constraints, Constraint, False, [c1, c2, c3]),
        (constraints, NChooseKConstraint, False, [c3]),
    ],
)
def test_constraints_get(constraints, ConstraintType, exact, expected):
    returned = constraints.get(ConstraintType, exact=exact).constraints
    assert returned == expected
    for i in range(len(expected)):
        assert id(expected[i]) == id(returned[i])


def test_constraints_plus():
    returned = constraints + constraints2
    assert returned.constraints == [c1, c2, c3, c4, c5]


@pytest.mark.parametrize(
    "constraints, num_candidates",
    [
        (constraints2, 5),
    ],
)
def test_constraints_call(constraints, num_candidates):
    candidates = input_features.sample(num_candidates, SamplingMethodEnum.UNIFORM)
    returned = constraints(candidates)
    assert returned.shape == (num_candidates, len(constraints))


@pytest.mark.parametrize(
    "constraints, num_candidates, fulfilled",
    [
        (constraints2, 5, False),
        (constraints3, 5, True),
    ],
)
def test_constraints_is_fulfilled(constraints, num_candidates, fulfilled):
    candidates = input_features.sample(num_candidates, SamplingMethodEnum.UNIFORM)
    returned = constraints.is_fulfilled(candidates)
    assert returned.shape == (num_candidates,)
    assert returned.dtype == bool
    assert returned.all() == fulfilled
