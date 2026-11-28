from bofire.data_models.strategies.api import (
    AlwaysTrueCondition,
    CombiCondition,
    DistinguishableParameters,
    FeasibleExperimentCondition,
    NumberOfExperimentsCondition,
    SufficientDistinguishableValues,
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
    SufficientDistinguishableValues,
    lambda: {
        "inputs": [
            DistinguishableParameters(
                key="x1", threshold=0.5, n_different_elements=3
            ).model_dump()
        ],
        "outputs": [
            DistinguishableParameters(
                key="y1", threshold=1.0, n_different_elements=2
            ).model_dump()
        ],
    },
)

specs.add_invalid(
    SufficientDistinguishableValues,
    dict,
    error=ValueError,
    message="At least one input or output must be listed.",
)
