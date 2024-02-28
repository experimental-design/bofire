import random

from pydantic.error_wrappers import ValidationError

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

for scale in [-1.0, 0.0]:
    specs.add_invalid(
        priors.NormalPrior,
        lambda: {
            "loc": random.random(),
            "scale": scale,  # noqa: B023
        },
        error=ValidationError,
    )

specs.add_valid(
    priors.GammaPrior,
    lambda: {
        "concentration": random.random(),
        "rate": random.random(),
    },
)

for rate in [-1.0, 0]:
    for concentration in [-5, 6]:
        specs.add_invalid(
            priors.GammaPrior,
            lambda: {
                "concentration": concentration,  # noqa: B023
                "rate": rate,  # noqa: B023
            },
            error=ValidationError,
        )
