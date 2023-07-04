from bofire.data_models.strategies.api import (
    AlwaysTrueCondition,
    CombiCondition,
    NumberOfExperimentsCondition,
)
from tests.bofire.data_models.specs.specs import Specs

specs = Specs([])

specs.add_valid(
    AlwaysTrueCondition,
    lambda: {},
)

specs.add_valid(
    NumberOfExperimentsCondition,
    lambda: {"n_experiments": 10},
)

specs.add_valid(
    CombiCondition,
    lambda: {
        "conditions": [
            NumberOfExperimentsCondition(n_experiments=5).dict(),
            AlwaysTrueCondition().dict(),
        ],
        "n_required_conditions": 2,
    },
)
