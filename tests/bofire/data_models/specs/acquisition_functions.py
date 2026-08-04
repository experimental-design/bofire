import random

import bofire.data_models.acquisition_functions.api as acquisition_functions
from tests.bofire.data_models.specs.specs import Specs


specs = Specs([])

specs.add_valid(
    acquisition_functions.qLogPF,
    lambda: {"n_mc_samples": 512},
)

specs.add_valid(
    acquisition_functions.qEI,
    lambda: {"n_mc_samples": 512},
)

specs.add_valid(
    acquisition_functions.qLogEI,
    lambda: {"n_mc_samples": 512},
)

specs.add_invalid(
    acquisition_functions.qLogEI,
    lambda: {"n_mc_samples": 513},
    error=ValueError,
    message="Argument is not power of two.",
)

specs.add_valid(
    acquisition_functions.qNEI,
    lambda: {"prune_baseline": random.choice([True, False]), "n_mc_samples": 512},
)

specs.add_valid(
    acquisition_functions.qLogNEI,
    lambda: {"prune_baseline": random.choice([True, False]), "n_mc_samples": 512},
)


specs.add_valid(
    acquisition_functions.qPI,
    lambda: {"tau": random.random(), "n_mc_samples": 512},
)

specs.add_valid(
    acquisition_functions.qSR,
    lambda: {"n_mc_samples": 512},
)

specs.add_valid(
    acquisition_functions.qUCB,
    lambda: {"beta": random.random(), "n_mc_samples": 512},
)

specs.add_valid(
    acquisition_functions.qEHVI,
    lambda: {"alpha": random.random(), "n_mc_samples": 512},
)

specs.add_valid(
    acquisition_functions.qLogEHVI,
    lambda: {"alpha": random.random(), "n_mc_samples": 512},
)

specs.add_valid(
    acquisition_functions.qNEHVI,
    lambda: {
        "alpha": random.random(),
        "prune_baseline": random.choice([True, False]),
        "n_mc_samples": 512,
    },
)

specs.add_valid(
    acquisition_functions.qLogNEHVI,
    lambda: {
        "alpha": random.random(),
        "prune_baseline": random.choice([True, False]),
        "n_mc_samples": 512,
    },
)

specs.add_valid(
    acquisition_functions.qNegIntPosVar,
    lambda: {
        "n_mc_samples": 128,
        "weights": {
            "y_1": 0.5,
            "y_2": 0.5,
        },
    },
)
