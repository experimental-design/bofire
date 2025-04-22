import importlib.util

import numpy as np
import pytest

from bofire.data_models.constraints.api import (
    LinearInequalityConstraint,
    NonlinearInequalityConstraint,
)
from bofire.data_models.domain.api import Domain
from bofire.data_models.enum import SamplingMethodEnum
from bofire.data_models.features.api import ContinuousInput, ContinuousOutput
from bofire.strategies.doe.doe_problem import (
    FirstOrderDoEProblem,
    SecondOrderDoEProblem,
)
from bofire.strategies.doe.objective import DOptimalityCriterion, get_objective_function
from bofire.strategies.doe.utils import (
    constraints_as_scipy_constraints,
    nchoosek_constraints_as_bounds,
)


CYIPOPT_AVAILABLE = importlib.util.find_spec("cyipopt") is not None


@pytest.mark.skipif(not CYIPOPT_AVAILABLE, reason="requires cyipopt")
def test_FirstOrderDoEProblem():
    n_experiments = 4
    criterion = DOptimalityCriterion(formula="linear")
    domain = Domain.from_lists(
        inputs=[
            ContinuousInput(key="x1", bounds=(0, 1)),
            ContinuousInput(key="x2", bounds=(0, 1)),
            ContinuousInput(key="x3", bounds=(0, 1)),
        ],
        outputs=[ContinuousOutput(key="y")],
        constraints=[
            NonlinearInequalityConstraint(
                expression="x1**2 + x2**2 - x3",
                features=["x1", "x2", "x3"],
                jacobian_expression="[2*x1,2*x2,-1]",
            ),
            LinearInequalityConstraint(
                features=["x1", "x2", "x3"],
                coefficients=[1, 1, 1],
                rhs=1,
            ),
        ],
    )

    objective_function = get_objective_function(
        criterion, domain=domain, n_experiments=n_experiments
    )
    assert objective_function is not None, "Criterion type is not supported!"

    x0 = (
        domain.inputs.sample(n=n_experiments, method=SamplingMethodEnum.UNIFORM)
        .to_numpy()
        .flatten()
    )

    constraints = constraints_as_scipy_constraints(
        domain,
        n_experiments,
        ignore_nchoosek=True,
    )

    bounds = nchoosek_constraints_as_bounds(domain, n_experiments)

    problem = FirstOrderDoEProblem(
        doe_objective=objective_function,
        bounds=bounds,
        constraints=constraints,
    )

    assert len(problem.linear_constraints) == 1
    assert len(problem.nonlinear_constraints) == 1
    assert problem.A.shape == (4, 12)
    assert np.allclose(problem.row, np.repeat([0, 1, 2, 3, 4, 5, 6, 7], 3))
    assert np.allclose(problem.col, np.tile([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], 2))

    with pytest.raises(AttributeError):
        problem.hessian(x0, [0.0, 0.0], 1.0)
    with pytest.raises(AttributeError):
        problem.hessianstructure()


@pytest.mark.skipif(not CYIPOPT_AVAILABLE, reason="requires cyipopt")
def test_SecondOrderDoEProblem():
    n_experiments = 4
    criterion = DOptimalityCriterion(formula="linear")
    domain = Domain.from_lists(
        inputs=[
            ContinuousInput(key="x1", bounds=(0, 1)),
            ContinuousInput(key="x2", bounds=(0, 1)),
            ContinuousInput(key="x3", bounds=(0, 1)),
        ],
        outputs=[ContinuousOutput(key="y")],
        constraints=[
            NonlinearInequalityConstraint(
                expression="x1**2 + x2**2 - x3",
                features=["x1", "x2", "x3"],
                jacobian_expression="[2*x1,2*x2,-1]",
            ),
            LinearInequalityConstraint(
                features=["x1", "x2", "x3"],
                coefficients=[1, 1, 1],
                rhs=1,
            ),
        ],
    )

    objective_function = get_objective_function(
        criterion, domain=domain, n_experiments=n_experiments
    )
    assert objective_function is not None, "Criterion type is not supported!"

    x0 = (
        domain.inputs.sample(n=n_experiments, method=SamplingMethodEnum.UNIFORM)
        .to_numpy()
        .flatten()
    )

    constraints = constraints_as_scipy_constraints(
        domain,
        n_experiments,
        ignore_nchoosek=True,
    )

    bounds = nchoosek_constraints_as_bounds(domain, n_experiments)

    problem = SecondOrderDoEProblem(
        doe_objective=objective_function,
        bounds=bounds,
        constraints=constraints,
    )

    assert len(problem.linear_constraints) == 1
    assert len(problem.nonlinear_constraints) == 1
    assert problem.A.shape == (4, 12)
    assert np.allclose(problem.row, np.repeat([0, 1, 2, 3, 4, 5, 6, 7], 3))
    assert np.allclose(problem.col, np.tile([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], 2))

    problem.hessian(x0, np.array([0.0, 0.0]), 1.0)
    problem.hessianstructure()
