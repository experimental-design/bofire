import importlib
import importlib.util

import numpy as np
import pandas as pd
import pytest

from bofire.data_models.constraints.api import NonlinearInequalityConstraint


SYMPY_AVAILABLE = importlib.util.find_spec("sympy") is not None
TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@pytest.mark.skipif(not SYMPY_AVAILABLE, reason="requires rdkit")
@pytest.mark.skipif(not TORCH_AVAILABLE, reason="requires torch")
def test_nonlinear_constraints_jacobian_expression():
    import torch

    constraint0 = NonlinearInequalityConstraint(
        expression="x1**2 + x2**2 - x3",
        features=["x1", "x2", "x3"],
    )
    constraint1 = NonlinearInequalityConstraint(
        expression="x1**2 + x2**2 - x3",
        features=["x1", "x2", "x3"],
        jacobian_expression="[2*x1, 2*x2, -1]",
    )

    data = pd.DataFrame(np.random.rand(10, 3), columns=["x1", "x2", "x3"])
    assert np.allclose(constraint0.jacobian(data), constraint1.jacobian(data))

    constraint2 = NonlinearInequalityConstraint(
        expression=lambda x1, x2, x3: x1**2 + x2**2 - x3,
        features=["x1", "x2", "x3"],
    )
    constraint3 = NonlinearInequalityConstraint(
        expression=lambda x1, x2, x3: x1**2 + x2**2 - x3,
        features=["x1", "x2", "x3"],
        jacobian_expression=lambda x1, x2, x3: [
            2 * x1,
            2 * x2,
            -1.0 * torch.ones_like(x3),
        ],
    )

    assert np.allclose(constraint2.jacobian(data), constraint0.jacobian(data))
    assert np.allclose(constraint3.jacobian(data), constraint0.jacobian(data))

    with pytest.raises(
        ValueError,
        match="Provided features do not match the features used in the expression.",
    ):
        NonlinearInequalityConstraint(
            expression=lambda x1, x2, x3: x1**2 + x2**2 - x3,
            features=["x1", "x2", "x5"],
            jacobian_expression=None,
        )


@pytest.mark.skipif(not SYMPY_AVAILABLE, reason="requires rdkit")
@pytest.mark.skipif(not TORCH_AVAILABLE, reason="requires torch")
def test_nonlinear_constraints_hessian_expression():
    import torch

    constraint0 = NonlinearInequalityConstraint(
        expression="x1**3 + x2**2 - x3",
        features=["x1", "x2", "x3"],
    )
    constraint1 = NonlinearInequalityConstraint(
        expression="x1**3 + x2**2 - x3",
        features=["x1", "x2", "x3"],
        jacobian_expression="[3*x1**2, 2*x2, -1]",
        hessian_expression="[[6*x1, 0, 0], [0, 2, 0], [0, 0, 0]]",
    )

    data = pd.DataFrame(np.random.rand(10, 3), columns=["x1", "x2", "x3"])
    for i in range(10):
        assert np.allclose(constraint0.hessian(data)[i], constraint1.hessian(data)[i])

    constraint2 = NonlinearInequalityConstraint(
        features=["x1", "x2", "x3"],
        expression=lambda x1, x2, x3: x1**3 + x2**2 - x3,
    )
    constraint3 = NonlinearInequalityConstraint(
        features=["x1", "x2", "x3"],
        expression=lambda x1, x2, x3: x1**3 + x2**2 - x3,
        jacobian_expression=lambda x1, x2, x3: [
            3 * x1**2,
            2 * x2,
            -1.0 * torch.ones_like(x3),
        ],
        hessian_expression=lambda x1, x2, x3: [
            [6 * x1, 0, 0],
            [0, 2, 0],
            [0, 0, 0],
        ],
    )

    for i in range(10):
        assert np.allclose(constraint2.hessian(data)[i], constraint0.hessian(data)[i])
        assert np.allclose(constraint3.hessian(data)[i], constraint0.hessian(data)[i])

    with pytest.raises(
        ValueError,
        match="Provided features do not match the features used in the expression.",
    ):
        NonlinearInequalityConstraint(
            expression=lambda x1, x2, x3: x1**2 + x2**2 - x3,
            features=["x1", "x2", "x5"],
            hessian_expression=None,
        )
