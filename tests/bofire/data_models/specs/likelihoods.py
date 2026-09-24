import math

import bofire.data_models.likelihoods.api as likelihoods
from bofire.data_models.priors.api import (
    HVARFNER_NOISE_PRIOR,
    THREESIX_NOISE_PRIOR,
    GreaterThan,
)
from tests.bofire.data_models.specs.specs import Specs


specs = Specs([])

specs.add_valid(
    likelihoods.GaussianLikelihood,
    lambda: {
        "noise_prior": HVARFNER_NOISE_PRIOR().model_dump(),
        "noise_constraint": GreaterThan(
            lower_bound=1e-4, initial_value=math.exp(-5.0)
        ).model_dump(),
    },
)
specs.add_valid(
    likelihoods.GaussianLikelihood,
    lambda: {
        "noise_prior": THREESIX_NOISE_PRIOR().model_dump(),
        "noise_constraint": None,
    },
)
