from bofire.data_models.strategies.api import TuRBOConfig
from tests.bofire.data_models.specs.specs import Specs


specs = Specs([])

specs.add_valid(
    TuRBOConfig,
    lambda: {
        "length_init": 0.8,
        "length_min": 1e-2,
        "length_max": 1.6,
        "lengthscale_adjustment_factor": 2.0,
        "fit_region_multiplier": 2.0,
        "min_tr_size": 10,
        "max_tr_size": 2048,
        "success_epsilon": 1e-3,
        "success_streak": 3,
        "failure_streak": 3,
        "success_counter": 0,
        "failure_counter": 0,
        "use_independent_tr": False,
        "length": 0.8,
        "X_center_idx": -1,
    },
)

specs.add_invalid(TuRBOConfig, lambda: {"length_min": -0.1}, error=ValueError)
specs.add_invalid(
    TuRBOConfig, lambda: {"lengthscale_adjustment_factor": 0.9}, error=ValueError
)
specs.add_invalid(TuRBOConfig, lambda: {"fit_region_multiplier": 0.9}, error=ValueError)
