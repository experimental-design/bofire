from bofire.data_models.strategies.api import (
    AlwaysTrueCondition,
    CombiCondition,
    FeasibleExperimentCondition,
    NumberOfExperimentsCondition,
    UCBLCBRegretBoundCondition,
)
from tests.bofire.data_models.specs.specs import Specs


specs = Specs([])

specs.add_valid(
    AlwaysTrueCondition,
    dict,
)

specs.add_valid(
    NumberOfExperimentsCondition,
    lambda: {"n_experiments": 10},
)

specs.add_valid(
    CombiCondition,
    lambda: {
        "conditions": [
            NumberOfExperimentsCondition(n_experiments=5).model_dump(),
            AlwaysTrueCondition().model_dump(),
        ],
        "n_required_conditions": 2,
    },
)

specs.add_valid(
    FeasibleExperimentCondition,
    lambda: {"n_required_feasible_experiments": 3, "threshold": 0.95},
)

specs.add_valid(
    UCBLCBRegretBoundCondition,
    lambda: {"noise_variance": 0.1, "threshold_factor": 2.0, "min_experiments": 10},
)

specs.add_valid(
    UCBLCBRegretBoundCondition,
    lambda: {
        "noise_variance": "cv",
        "threshold_factor": 0.5,
        "cv_fold_columns": ["f0", "f1", "f2", "f3", "f4"],
        "min_experiments": 5,
    },
)
