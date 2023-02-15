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
from tests.bofire import specs


def test_valid_constraint_specs(valid_constraint_spec: specs.Spec):
    res = valid_constraint_spec.obj()
    assert isinstance(res, valid_constraint_spec.cls)
    assert isinstance(res.__str__(), str)


def test_invalid_constraint_specs(invalid_constraint_spec: specs.Spec):
    with pytest.raises((ValueError, TypeError, KeyError, ValidationError)):
        invalid_constraint_spec.obj()


def test_from_greater_equal():
    spec = specs.constraints.valid(LinearInequalityConstraint).spec
    c = LinearInequalityConstraint.from_greater_equal(
        features=spec["features"],
        coefficients=spec["coefficients"],
        rhs=spec["rhs"],
    )
    assert c.rhs == spec["rhs"] * -1.0
    assert c.coefficients == [-1.0 * coef for coef in spec["coefficients"]]
    assert c.features == spec["features"]


def test_as_greater_equal():
    spec = specs.constraints.valid(LinearInequalityConstraint).spec
    c = LinearInequalityConstraint.from_greater_equal(
        features=spec["features"],
        coefficients=spec["coefficients"],
        rhs=spec["rhs"],
    )
    features, coefficients, rhs = c.as_greater_equal()
    assert c.rhs == rhs * -1.0
    assert coefficients == [-1.0 * coef for coef in c.coefficients]
    assert c.features == features


def test_from_smaller_equal():
    spec = specs.constraints.valid(LinearInequalityConstraint).spec
    c = LinearInequalityConstraint.from_smaller_equal(
        features=spec["features"],
        coefficients=spec["coefficients"],
        rhs=spec["rhs"],
    )
    assert c.rhs == spec["rhs"]
    assert c.coefficients == spec["coefficients"]
    assert c.features == spec["features"]


def test_as_smaller_equal():
    spec = specs.constraints.valid(LinearInequalityConstraint).spec
    c = LinearInequalityConstraint.from_smaller_equal(
        features=spec["features"],
        coefficients=spec["coefficients"],
        rhs=spec["rhs"],
    )
    features, coefficients, rhs = c.as_smaller_equal()
    assert c.rhs == rhs
    assert coefficients == c.coefficients
    assert c.features == features


# test the Constraints Class
c1 = specs.constraints.valid(LinearEqualityConstraint).obj()
c2 = specs.constraints.valid(LinearInequalityConstraint).obj()
c3 = specs.constraints.valid(NChooseKConstraint).obj()
c4 = specs.constraints.valid(NonlinearEqualityConstraint).obj()
c5 = specs.constraints.valid(NonlinearInequalityConstraint).obj()
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


@pytest.mark.parametrize(
    "constraints",
    [
        (["s"]),
        ([specs.constraints.valid(LinearInequalityConstraint).obj()], 5),
        (
            [specs.constraints.valid(LinearInequalityConstraint).obj()],
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
