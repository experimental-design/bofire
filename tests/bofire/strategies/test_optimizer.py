from typing import List

import numpy as np
import pandas as pd
import pytest
import torch
from pymoo.optimize import minimize as pymoo_minimize

from bofire.data_models.strategies import api as data_models_strategies
from bofire.strategies.utils import get_ga_problem_and_algorithm
from bofire.surrogates import api as bofire_surrogates


@pytest.fixture(
    params=[  # (optimizer data model, params)
        data_models_strategies.BotorchOptimizer(),
        data_models_strategies.GeneticAlgorithmOptimizer(
            population_size=100, n_max_gen=100
        ),
    ]
)
def optimizer_data_model(request) -> data_models_strategies.AcquisitionOptimizer:
    return request.param


def test_optimizer(optimizer_benchmark, optimizer_data_model):
    # sort out cases where the optimizer does not support nonlinear constraints

    strategy = optimizer_benchmark.get_strategy(optimizer_data_model)

    np.random.seed(42)  # for reproducibility
    proposals = strategy.ask(optimizer_benchmark.n_add)

    assert proposals.shape[0] == optimizer_benchmark.n_add

    constraints = strategy.domain.constraints.get()
    for constr in constraints.constraints:
        assert constr.is_fulfilled(proposals, tol=1e-4).all()


def test_torch_objective_function(optimizer_benchmark, optimizer_data_model):
    # sort out cases where the optimizer does not support nonlinear constraints
    if isinstance(optimizer_data_model, data_models_strategies.BotorchOptimizer):
        pytest.skip("skipping multi-objective optimization for botorch optimizer")

    # we get the strategy object  for the input-preprocessing specs, and the surrogates
    strategy = optimizer_benchmark.get_strategy(optimizer_data_model)

    surrogates: bofire_surrogates.BotorchSurrogates = strategy.surrogates

    q = 1

    def objective_function(x: torch.Tensor) -> torch.Tensor:
        """Objective function that evaluates the mean valuew of the surrogates at the given input x."""
        y = torch.hstack(
            [
                sg.model.posterior(x).mean.reshape((-1, q))
                for sg in surrogates.surrogates
            ]
        )
        return y

    problem, algorithm, termination = get_ga_problem_and_algorithm(
        optimizer_data_model,
        strategy.domain,
        [objective_function],
        q=q,
        callable_format="torch",
        n_obj=len(surrogates.surrogates) * q,
        verbose=True,
    )

    _ = pymoo_minimize(problem, algorithm, termination, verbose=True)


def test_pandas_objective_function(optimizer_benchmark, optimizer_data_model):
    # sort out cases where the optimizer does not support nonlinear constraints
    if isinstance(optimizer_data_model, data_models_strategies.BotorchOptimizer):
        pytest.skip("skipping multi-objective optimization for botorch optimizer")

    domain = optimizer_benchmark.benchmark.domain

    def objective_function(x: List[pd.DataFrame]) -> np.ndarray:
        """assume we want to maximize the mean variance of the experiments dataframe"""

        vars = [xi.var(numeric_only=True).mean() for xi in x]
        return np.array(vars)

    problem, algorithm, termination = get_ga_problem_and_algorithm(
        optimizer_data_model,
        domain,
        [objective_function],
        q=optimizer_benchmark.n_add,
        callable_format="pandas",
        n_obj=1,
        verbose=True,
    )

    _ = pymoo_minimize(problem, algorithm, termination, verbose=True)
