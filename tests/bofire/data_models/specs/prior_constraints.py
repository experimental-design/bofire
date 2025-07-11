import bofire.data_models.priors.api as priors
from tests.bofire.data_models.specs.specs import Specs


specs = Specs([])

specs.add_valid(
    priors.NonTransformedInterval,
    lambda: {
        "lower_bound": 0.05,
        "upper_bound": float("inf"),
        "initial_value": 0.2,
    },
)

specs.add_valid(
    priors.NonTransformedInterval,
    lambda: {
        "lower_bound": 0.01,
        "upper_bound": 10,
        "initial_value": 0.1,
    },
)
