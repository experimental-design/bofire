from pydantic.utils import deep_update

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
