import math

from pydantic import ValidationError

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
specs.add_valid(
    likelihoods.SparseOutlierGaussianLikelihood,
    lambda: {
        "noise_prior": HVARFNER_NOISE_PRIOR().model_dump(),
        "noise_constraint": GreaterThan(lower_bound=1e-4).model_dump(),
        "prior_mean_of_support": 3,
        "convex_parametrization": False,
        "cache_model_trace": True,
    },
)
specs.add_valid(
    likelihoods.TaskwiseLikelihood,
    lambda: {
        "likelihood": likelihoods.GaussianLikelihood(
            noise_prior=HVARFNER_NOISE_PRIOR()
        ).model_dump(),
    },
)

specs.add_invalid(
    likelihoods.TaskwiseLikelihood,
    lambda: {
        "likelihood": likelihoods.SparseOutlierGaussianLikelihood(
            prior_mean_of_support=3
        ),
    },
    error=ValidationError,
)
specs.add_invalid(
    likelihoods.TaskwiseLikelihood,
    lambda: {
        "likelihood": likelihoods.SparseOutlierGaussianLikelihood(
            prior_mean_of_support=3
        ).model_dump(),
    },
    error=ValidationError,
)


specs.add_valid(likelihoods.PairwiseProbitLikelihood, lambda: {})
specs.add_valid(likelihoods.PairwiseLogitLikelihood, lambda: {})
