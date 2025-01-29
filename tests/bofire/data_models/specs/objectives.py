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
    lambda: {"w": 1.0, "bounds": [0.1, 0.9]},
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
    lambda: {"w": 1.0, "bounds": [0.1, 0.9]},
)

specs.add_valid(
    objectives.MovingMaximizeSigmoidObjective,
    lambda: {"w": 1.0, "tp": 0.3, "steepness": 0.2},
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

for obj in [
    objectives.IncreasingDesirabilityObjective,
    objectives.DecreasingDesirabilityObjective,
]:
    specs.add_valid(
        obj,
        lambda: {"w": 1.0, "bounds": [0, 10.0], "log_shape_factor": 1.0, "clip": True},
    )
    specs.add_valid(
        obj,
        lambda: {"w": 1.0, "bounds": [0, 10.0], "log_shape_factor": -1.0, "clip": True},
    )
    specs.add_invalid(
        obj,
        lambda: {
            "w": 1.0,
            "bounds": [0, 10.0],
            "log_shape_factor": -1.0,
            "clip": False,
        },
        ValueError,
        "Log shape factor log_shape_factor must be zero if clip is False.",
    )

specs.add_valid(
    objectives.PeakDesirabilityObjective,
    lambda: {
        "w": 1.0,
        "bounds": [0, 10.0],
        "clip": True,
        "log_shape_factor": 0.0,
        "log_shape_factor_decreasing": 0.0,
        "peak_position": 5.0,
    },
)
specs.add_invalid(
    objectives.PeakDesirabilityObjective,
    lambda: {
        "w": 1.0,
        "bounds": [0, 10.0],
        "clip": False,
        "log_shape_factor": 0.0,
        "log_shape_factor_decreasing": 1.0,
        "peak_position": 5.0,
    },
    ValueError,
    "Log shape factor log_shape_factor_decreasing must be zero if clip is False.",
)
specs.add_invalid(
    objectives.PeakDesirabilityObjective,
    lambda: {"bounds": [0, 10.0], "peak_position": 15.0},
    ValueError,
    "Peak position must be within bounds",
)
specs.add_invalid(
    objectives.PeakDesirabilityObjective,
    lambda: {"bounds": [0, 10.0], "peak_position": -1.0},
    ValueError,
    "Peak position must be within bounds",
)
specs.add_valid(
    objectives.InRangeDesirability,
    lambda: {
        "bounds": [0.0, 10.0],
        "clip": True,
        "w": 1.0,
    },
)
