from bofire.data_models.strategies.api import (
    AlwaysTrueCondition,
    CombiCondition,
    ExpMinRegretGapCondition,
    FeasibleExperimentCondition,
    LogEIPCCondition,
    NumberOfExperimentsCondition,
    ProbabilisticRegretBoundCondition,
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
    lambda: {
        "noise_variance": 0.1,
        "threshold_factor": 2.0,
        "cv_fold_columns": None,
        "topq": 1.0,
        "min_topq": 20,
        "min_experiments": 10,
        "delta": 0.1,
        "beta_scale": 0.2,
        "n_samples_lcb": 2000,
        "batch_size": 512,
        "lcb_method": "sample",
        "fallback_noise_variance": 1e-4,
    },
)

specs.add_valid(
    UCBLCBRegretBoundCondition,
    lambda: {
        "noise_variance": "cv",
        "threshold_factor": 0.5,
        "cv_fold_columns": ["f0", "f1", "f2", "f3", "f4"],
        "topq": 1.0,
        "min_topq": 20,
        "min_experiments": 5,
        "delta": 0.05,
        "beta_scale": 1.0,
        "n_samples_lcb": 500,
        "batch_size": 256,
        "lcb_method": "optimize",
        "fallback_noise_variance": 1e-6,
    },
)

specs.add_valid(
    ExpMinRegretGapCondition,
    lambda: {
        "threshold_mode": "adaptive",
        "delta": 0.1,
        "rate": 0.1,
        "start_timing": 10,
        "min_experiments": 5,
        "beta_scale": 1.0,
        "n_samples_lcb": 1000,
    },
)

specs.add_valid(
    ExpMinRegretGapCondition,
    lambda: {
        "threshold_mode": "median",
        "delta": 0.1,
        "rate": 0.05,
        "start_timing": 15,
        "min_experiments": 5,
        "beta_scale": 1.0,
        "n_samples_lcb": 1000,
    },
)

specs.add_valid(
    LogEIPCCondition,
    lambda: {
        "lambda_cost": 1.0,
        "cost_column": None,
        "cost_value": 1.0,
        "alpha": 1.0,
        "min_experiments": 5,
        "n_samples": 2000,
        "search_method": "sample",
        "cost_model": "mean",
    },
)

specs.add_valid(
    ProbabilisticRegretBoundCondition,
    lambda: {
        "epsilon": None,
        "epsilon_relative": 0.01,
        "delta_mod": 0.05,
        "delta_est": 0.05,
        "optim_method": "L-BFGS-B",
        "optim_maxiter": 200,
        "optim_ftol": 1e-09,
        "enforce_convergence": True,
        "n_samples_max": 1024,
        "initial_batch": 16,
        "batch_growth": 1.5,
        "min_experiments": 5,
        "n_starts": 8,
        "n_random": 512,
        "n_test_points": 1,
    },
)
