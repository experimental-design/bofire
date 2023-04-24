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

specs.add_invalid(
    objectives.MinimizeObjective,
    lambda: {"w": 1.0, "lower_bound": 0.1, "upper_bound": 0.9},
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
