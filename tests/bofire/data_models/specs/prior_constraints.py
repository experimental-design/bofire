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

specs.add_valid(
    priors.LogTransformedInterval,
    lambda: {
        "lower_bound": 0.05,
        "upper_bound": 1,
        "initial_value": 0.2,
    },
)

specs.add_invalid(
    priors.NonTransformedInterval,
    lambda: {
        "lower_bound": 2,
        "upper_bound": 1,
        "initial_value": 1.5,
    },
    error=ValueError,
    message="The lower bound must be less than the upper bound for an interval.",
)

specs.add_invalid(
    priors.NonTransformedInterval,
    lambda: {
        "lower_bound": 1,
        "upper_bound": 10,
        "initial_value": 11,
    },
    error=ValueError,
    message="The initial value must be within the bounds of the interval.",
)

specs.add_valid(priors.Positive, lambda: {})
specs.add_valid(priors.GreaterThan, lambda: {"lower_bound": 42})
specs.add_valid(priors.LessThan, lambda: {"upper_bound": 42})
