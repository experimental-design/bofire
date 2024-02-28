import random

import numpy as np
import pytest
from pydantic.utils import deep_update

import bofire.data_models.strategies.api as data_models
from bofire.benchmarks.api import Hartmann
from bofire.data_models.features.api import Input
from bofire.strategies.api import EntingStrategy
from tests.bofire.strategies.test_base import domains

VALID_ENTING_STRATEGY_SPEC = {
    "domain": domains[1],
    "enting_params": {"unc_params": {"dist_metric": "l1", "acq_sense": "penalty"}},
}

ENTING_STRATEGY_SPECS = {
    "valids": [
        VALID_ENTING_STRATEGY_SPEC,
        {
            **VALID_ENTING_STRATEGY_SPEC,
            "enting_params": {"tree_train_params": {"train_lib": "lgbm"}},
        },
    ],
    "invalids": [
        deep_update(
            VALID_ENTING_STRATEGY_SPEC,
            {"enting_params": {"unc_params": {"acq_sense": None}}},
        ),
        deep_update(
            VALID_ENTING_STRATEGY_SPEC,
            {"enting_params": {"unc_params": {"distance_metric": None}}},
        ),
        deep_update(
            VALID_ENTING_STRATEGY_SPEC,
            {"enting_params": {"tree_train_params": {"train_lib": None}}},
        ),
        deep_update(
            VALID_ENTING_STRATEGY_SPEC,
            {
                "enting_params": {
                    "tree_train_params": {"train_params": {"max_depth": -3}}
                }
            },
        ),
    ],
}

SOLVER_PARAMS = {"solver_name": "gurobi", "verbose": False}


@pytest.mark.parametrize(
    "domain, enting_params, solver_params",
    [
        (
            domains[0],
            VALID_ENTING_STRATEGY_SPEC["enting_params"],
            SOLVER_PARAMS,
        )
    ],
)
def test_enting_not_fitted(domain, enting_params, solver_params):
    data_model = data_models.EntingStrategy(
        domain=domain, enting_params=enting_params, solver_params=solver_params
    )
    strategy = EntingStrategy(data_model=data_model)

    msg = "Uncertainty model needs fit function call before it can predict."
    with pytest.raises(AssertionError, match=msg):
        strategy._ask(1)


@pytest.mark.parametrize(
    "domain, enting_params, solver_params",
    [
        (
            domains[0],
            VALID_ENTING_STRATEGY_SPEC["enting_params"],
            SOLVER_PARAMS,
        )
    ],
)
def test_enting_param_consistency(domain, enting_params, solver_params):
    # compare EntingParams objects between entmoot and bofire
    data_model = data_models.EntingStrategy(
        domain=domain, enting_params=enting_params, solver_params=solver_params
    )
    strategy = EntingStrategy(data_model=data_model)
    assert strategy._enting._acq_sense == data_model.enting_params.unc_params.acq_sense
    assert strategy._enting._beta == data_model.enting_params.unc_params.beta


@pytest.mark.parametrize(
    "allowed_k",
    [1, 3, 5, 6],
)
@pytest.mark.slow
def test_nchoosek_constraint_with_enting(allowed_k):
    benchmark = Hartmann(6, allowed_k=allowed_k)
    samples = benchmark.domain.inputs.sample(10)
    experiments = benchmark.f(samples, return_complete=True)

    enting_params = VALID_ENTING_STRATEGY_SPEC["enting_params"]
    solver_params = SOLVER_PARAMS
    data_model = data_models.EntingStrategy(
        domain=benchmark.domain,
        enting_params=enting_params,
        solver_params=solver_params,
    )
    strategy = EntingStrategy(data_model)

    strategy.tell(experiments)
    proposal = strategy.ask(1)

    input_values = proposal[benchmark.domain.get_feature_keys(Input)]
    assert (input_values != 0).sum().sum() == allowed_k


@pytest.mark.slow
def test_propose_optimal_point():
    # regression test, ensure that a good point is proposed
    np.random.seed(42)
    random.seed(42)

    benchmark = Hartmann(6)
    samples = benchmark.domain.inputs.sample(50)
    experiments = benchmark.f(samples, return_complete=True)

    enting_params = VALID_ENTING_STRATEGY_SPEC["enting_params"]
    solver_params = SOLVER_PARAMS
    data_model = data_models.EntingStrategy(
        domain=benchmark.domain,
        enting_params=enting_params,
        solver_params=solver_params,
    )
    strategy = EntingStrategy(data_model)

    # filter experiments to remove those in a box surrounding optimum
    radius = 0.5
    X = experiments[benchmark.domain.get_feature_keys(Input)].values
    X_opt = benchmark.get_optima()[benchmark.domain.get_feature_keys(Input)].values
    l1_dist_to_optimum = ((X - X_opt) ** 2).sum(axis=1)
    include = l1_dist_to_optimum > radius

    strategy.tell(experiments[include])
    proposal = strategy.ask(1)

    assert np.allclose(
        proposal.loc[0, benchmark.domain.get_feature_keys(Input)].tolist(),
        [0.275, 0.95454, 0.28484, 0.34005, 0.015457, 0.06135],
        atol=1e-6,
    )
