import random

import bofire.data_models.priors.api as priors
from tests.bofire.data_models.specs.specs import Specs

specs = Specs([])

specs.add_valid(
    priors.NormalPrior,
    lambda: {
        "loc": random.random(),
        "scale": random.random(),
    },
)
specs.add_valid(
    priors.GammaPrior,
    lambda: {
        "concentration": random.random(),
        "rate": random.random(),
    },
)
