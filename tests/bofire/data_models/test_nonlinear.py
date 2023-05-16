import numpy as np
import pandas as pd

from bofire.data_models.constraints.api import NonlinearInequalityConstraint


def test_nonlinear_constraints_jacobian_expression():
    constraint0 = NonlinearInequalityConstraint(
        expression="x1**2 + x2**2 - x3", features=["x1", "x2", "x3"]
    )
    constraint1 = NonlinearInequalityConstraint(
        expression="x1**2 + x2**2 - x3",
        features=["x1", "x2", "x3"],
        jacobian_expression="[2*x1, 2*x2, -1]",
    )

    data = pd.DataFrame(np.random.rand(10, 3), columns=["x1", "x2", "x3"])
    assert np.allclose(constraint0.jacobian(data), constraint1.jacobian(data))
