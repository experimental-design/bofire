import random

import bofire.data_models.acquisition_functions.api as acquisition_functions
from tests.bofire.data_models.specs.specs import Specs

specs = Specs([])

specs.add_valid(
    acquisition_functions.qEI,
    lambda: {},
)

specs.add_valid(
    acquisition_functions.qLogEI,
    lambda: {},
)

specs.add_valid(
    acquisition_functions.qNEI,
    lambda: {"prune_baseline": random.choice([True, False])},
)

specs.add_valid(
    acquisition_functions.qLogNEI,
    lambda: {"prune_baseline": random.choice([True, False])},
)


specs.add_valid(
    acquisition_functions.qPI,
    lambda: {
        "tau": random.random(),
    },
)

specs.add_valid(
    acquisition_functions.qSR,
    lambda: {},
)

specs.add_valid(
    acquisition_functions.qUCB,
    lambda: {
        "beta": random.random(),
    },
)

specs.add_valid(
    acquisition_functions.qEHVI,
    lambda: {
        "alpha": random.random(),
    },
)

specs.add_valid(
    acquisition_functions.qLogEHVI,
    lambda: {
        "alpha": random.random(),
    },
)

specs.add_valid(
    acquisition_functions.qNEHVI,
    lambda: {"alpha": random.random(), "prune_baseline": random.choice([True, False])},
)

specs.add_valid(
    acquisition_functions.qLogNEHVI,
    lambda: {"alpha": random.random(), "prune_baseline": random.choice([True, False])},
)
