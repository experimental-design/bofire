import numpy as np
import pandas as pd

import tests.bofire.data_models.specs.api as specs
from bofire.data_models.constraints.api import (
    LinearEqualityConstraint,
    LinearInequalityConstraint,
)


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


def test_hessian():
    spec = specs.constraints.valid(LinearInequalityConstraint).spec()
    c = LinearInequalityConstraint.from_smaller_equal(
        features=spec["features"],
        coefficients=spec["coefficients"],
        rhs=spec["rhs"],
    )
    experiments = pd.DataFrame(
        np.random.rand(10, len(spec["features"])), columns=spec["features"]
    )

    for i, key in enumerate(c.hessian(experiments).keys()):
        assert key == i
    for v in c.hessian(experiments).values():
        assert v == 0.0


def test_jacobian():
    spec = specs.constraints.valid(LinearInequalityConstraint).spec()
    c = LinearInequalityConstraint.from_smaller_equal(
        features=spec["features"],
        coefficients=spec["coefficients"],
        rhs=spec["rhs"],
    )
    experiments = pd.DataFrame(
        np.random.rand(10, len(spec["features"])), columns=spec["features"]
    )

    res = c.jacobian(experiments)
    assert list(res.columns) == [f"dg/d{name}" for name in spec["features"]]
    for i, idx in enumerate(res.index):
        assert idx == i
    assert res.shape == (experiments.shape[0], len(spec["features"]))
    for i in range(experiments.shape[0]):
        np.allclose(
            res.loc[i],
            np.array(spec["coefficients"])
            / np.linalg.norm(np.array(spec["coefficients"])),
        )


def test_linear_equality_to_description():
    c = LinearEqualityConstraint(
        features=["x1", "x2"], coefficients=[1.0, 2.0], rhs=5.0
    )
    assert c.to_description() == "1.0*x1 + 2.0*x2 = 5.0"


def test_linear_inequality_to_description():
    c = LinearInequalityConstraint(
        features=["x1", "x2"], coefficients=[1.0, 2.0], rhs=5.0
    )
    assert c.to_description() == "1.0*x1 + 2.0*x2 <= 5.0"


def test_linear_constraint_to_description_with_context():
    c = LinearInequalityConstraint(
        features=["x1", "x2"],
        coefficients=[1.0, 2.0],
        rhs=5.0,
        context="Safety limit",
    )
    assert c.to_description() == "1.0*x1 + 2.0*x2 <= 5.0 — Safety limit"
