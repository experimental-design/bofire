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
        lambda scale=scale: {
            "loc": random.random(),
            "scale": scale,
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
            lambda concentration=concentration, rate=rate: {
                "concentration": concentration,
                "rate": rate,
            },
            error=ValidationError,
        )

specs.add_valid(
    priors.LKJPrior,
    lambda: {
        "n_tasks": random.randint(2, 10),
        "shape": random.random(),
        "sd_prior": {
            "type": "GammaPrior",
            "concentration": random.random(),
            "rate": random.random(),
        },
    },
)

for shape in [-1, 0]:
    specs.add_invalid(
        priors.LKJPrior,
        lambda shape=shape: {
            "n_tasks": random.randint(1, 10),
            "shape": shape,
            "sd_prior": {
                "type": "GammaPrior",
                "concentration": random.random(),
                "rate": random.random(),
            },
        },
        error=ValidationError,
    )

for concentration in [-1, 0]:
    for rate in [-1, 0]:
        specs.add_invalid(
            priors.LKJPrior,
            lambda concentration=concentration, rate=rate: {
                "n_tasks": random.randint(1, 10),
                "shape": random.random(),
                "sd_prior": {
                    "type": "GammaPrior",
                    "concentration": concentration,
                    "rate": rate,
                },
            },
            error=ValidationError,
        )


specs.add_valid(
    priors.LogNormalPrior,
    lambda: {"loc": random.random(), "scale": random.random()},
)


specs.add_valid(
    priors.DimensionalityScaledLogNormalPrior,
    lambda: {
        "loc": random.random(),
        "loc_scaling": random.random(),
        "scale": random.random(),
        "scale_scaling": random.random(),
    },
)
