import bofire.data_models.objectives.api as objectives
from tests.bofire.data_models.specs.specs import Specs

specs = Specs([])

specs.add_valid(
    objectives.CloseToTargetObjective,
    lambda: {
        "target_value": 42,
        "exponent": 2,
        "w": 1.0,
    },
)
specs.add_valid(
    objectives.MaximizeObjective,
    lambda: {"w": 1.0, "bounds": (0.1, 0.9)},
)
specs.add_valid(
    objectives.MaximizeSigmoidObjective,
    lambda: {
        "steepness": 0.2,
        "tp": 0.3,
        "w": 1.0,
    },
)
specs.add_valid(
    objectives.MinimizeObjective,
    lambda: {"w": 1.0, "bounds": (0.1, 0.9)},
)


specs.add_valid(
    objectives.MinimizeSigmoidObjective,
    lambda: {
        "steepness": 0.2,
        "tp": 0.3,
        "w": 1.0,
    },
)
specs.add_valid(
    objectives.TargetObjective,
    lambda: {
        "w": 1.0,
        "target_value": 0.4,
        "tolerance": 0.4,
        "steepness": 0.3,
    },
)

specs.add_valid(
    objectives.ConstrainedCategoricalObjective,
    lambda: {
        "w": 1.0,
        "categories": ["green", "red", "blue"],
        "desirability": [True, False, True],
    },
)

specs.add_invalid(
    objectives.ConstrainedCategoricalObjective,
    lambda: {
        "w": 1.0,
        "categories": ["green", "red", "blue"],
        "desirability": [True, False, True, False],
    },
    error=ValueError,
    message="number of categories differs from number of desirabilities",
)

specs.add_invalid(
    objectives.ConstrainedCategoricalObjective,
    lambda: {
        "w": 1.0,
        "categories": ["green", "red", "blue", "blue"],
        "desirability": [True, False, True, False],
    },
    error=ValueError,
    message="Categories must be unique",
)
