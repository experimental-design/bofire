import pytest
from pydantic.utils import deep_update

import bofire.data_models.strategies.api as data_models
from bofire.strategies.api import EntingStrategy
from tests.bofire.strategies.test_base import domains

VALID_ENTING_STRATEGY_SPEC = {
    "domain": domains[1],
    "enting_params": {"unc_params": {"dist_metric": "l1", "acq_sense": "exploration"}},
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


@pytest.mark.parametrize(
    "domain, enting_params, solver_params",
    [
        (
            domains[0],
            VALID_ENTING_STRATEGY_SPEC["enting_params"],
            {"solver_name": "gurobi"},
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
