import numpy as np
import pandas as pd
import pytest
from pydantic.error_wrappers import ValidationError

import tests.bofire.data_models.specs.api as specs
from bofire.data_models.constraints.api import (
    Constraint,
    LinearConstraint,
    LinearEqualityConstraint,
    LinearInequalityConstraint,
    NChooseKConstraint,
    NonlinearConstraint,
    NonlinearEqualityConstraint,
    NonlinearInequalityConstraint,
)
from bofire.data_models.domain.api import Constraints, Inputs
from bofire.data_models.enum import SamplingMethodEnum
from bofire.data_models.features.api import ContinuousInput, ContinuousOutput


def test_from_greater_equal():
    spec = specs.constraints.valid(LinearInequalityConstraint).spec()
    c = LinearInequalityConstraint.from_greater_equal(
        features=spec["features"],
        coefficients=spec["coefficients"],
        rhs=spec["rhs"],
    )
    assert c.rhs == spec["rhs"] * -1.0
    assert c.coefficients == [-1.0 * coef for coef in spec["coefficients"]]
    assert c.features == spec["features"]


def test_as_greater_equal():
    spec = specs.constraints.valid(LinearInequalityConstraint).spec()
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
    spec = specs.constraints.valid(LinearInequalityConstraint).spec()
    c = LinearInequalityConstraint.from_smaller_equal(
        features=spec["features"],
        coefficients=spec["coefficients"],
        rhs=spec["rhs"],
    )
    assert c.rhs == spec["rhs"]
    assert c.coefficients == spec["coefficients"]
    assert c.features == spec["features"]


def test_as_smaller_equal():
    spec = specs.constraints.valid(LinearInequalityConstraint).spec()
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

if1 = ContinuousInput(key="f1", bounds=(0, 2))
if2 = ContinuousInput(key="f2", bounds=(0, 4))
if3 = ContinuousInput(key="f3", bounds=(3, 8))

inputs = Inputs(features=[if1, if2, if3])

constraints = Constraints(constraints=[c1, c2])
constraints2 = Constraints(constraints=[c4, c5])
constraints3 = Constraints(constraints=[c6])
constraints4 = Constraints(constraints=[c3])


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
        (constraints + constraints4, LinearConstraint, True, []),
        (constraints + constraints4, LinearConstraint, False, [c1, c2]),
        (constraints + constraints4, Constraint, False, [c1, c2, c3]),
        (constraints + constraints4, NChooseKConstraint, False, [c3]),
    ],
)
def test_constraints_get(constraints, ConstraintType, exact, expected):
    returned = constraints.get(ConstraintType, exact=exact).constraints
    assert returned == expected
    for i in range(len(expected)):
        assert id(expected[i]) == id(returned[i])


def test_constraints_plus():
    returned = constraints + constraints4 + constraints2
    assert returned.constraints == [c1, c2, c3, c4, c5]


@pytest.mark.parametrize(
    "constraints, num_candidates",
    [
        (constraints2, 5),
        (constraints4, 5),
    ],
)
def test_constraints_call(constraints, num_candidates):
    candidates = inputs.sample(num_candidates, SamplingMethodEnum.UNIFORM)
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
    candidates = inputs.sample(num_candidates, SamplingMethodEnum.UNIFORM)
    returned = constraints.is_fulfilled(candidates)
    assert returned.shape == (num_candidates,)
    assert returned.dtype == bool
    assert returned.all() == fulfilled


@pytest.mark.parametrize(
    "constraints, num_candidates",
    [
        (constraints, 2),
        (constraints2, 2),
    ],
)
def test_constraints_jacobian(constraints, num_candidates):
    candidates = inputs.sample(num_candidates, SamplingMethodEnum.UNIFORM)
    returned = constraints.jacobian(candidates)
    assert np.all(
        [
            list(returned[i].columns) == ["dg/df1", "dg/df2", "dg/df3"]
            for i, c in enumerate(constraints)
        ]
    )
    assert np.all(
        [
            returned[i].shape == (num_candidates, len(inputs))
            for i, c in enumerate(constraints)
        ]
    )
    for i, c in enumerate(constraints):
        if isinstance(c, LinearConstraint):
            assert np.allclose(
                returned[i],
                np.tile(
                    c.coefficients / np.linalg.norm(c.coefficients), (num_candidates, 1)
                ),
            )
        if isinstance(c, NonlinearConstraint):
            res = candidates.eval(c.jacobian_expression)
            for j, col in enumerate(res):
                if not hasattr(col, "__iter__"):
                    res[j] = pd.Series(np.repeat(col, candidates.shape[0]))
            assert np.allclose(returned[i], pd.DataFrame(res).transpose())
