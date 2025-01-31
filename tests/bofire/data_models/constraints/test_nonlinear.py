import importlib

import numpy as np
import pandas as pd
import pytest
import torch

from bofire.data_models.constraints.api import NonlinearInequalityConstraint


SYMPY_AVAILABLE = importlib.util.find_spec("sympy") is not None


@pytest.mark.skipif(not SYMPY_AVAILABLE, reason="requires rdkit")
def test_nonlinear_constraints_jacobian_expression():
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
    )
    constraint3 = NonlinearInequalityConstraint(
        expression=lambda x1, x2, x3: x1**2 + x2**2 - x3,
        jacobian_expression=lambda x1, x2, x3: [
            2 * x1,
            2 * x2,
            -1.0 * torch.ones_like(x3),
        ],
    )

    assert np.allclose(constraint2.jacobian(data), constraint0.jacobian(data))
    assert np.allclose(constraint3.jacobian(data), constraint0.jacobian(data))

    with pytest.raises(ValueError):
        NonlinearInequalityConstraint(
            expression=lambda x1, x2, x3: x1**2 + x2**2 - x3,
            features=["x1", "x2", "x3"],
            jacobian_expression=None,
        )
