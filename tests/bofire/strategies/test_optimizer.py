import numpy as np
import pytest

from bofire.data_models.strategies import api as data_models_strategies


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
        assert constr.is_fulfilled(proposals).all()
