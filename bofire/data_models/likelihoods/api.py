import math

from bofire.data_models.likelihoods.likelihood import GaussianLikelihood, Likelihood
from bofire.data_models.priors.api import (
    HVARFNER_NOISE_PRIOR,
    MBO_NOISE_PRIOR,
    THREESIX_NOISE_PRIOR,
    GreaterThan,
)
from bofire.data_models.unions import tagged_union


AnyLikelihood = tagged_union(GaussianLikelihood)


# likelihoods of interest, built anew on every call so that no two models share them
def THREESIX_LIKELIHOOD() -> GaussianLikelihood:
    """Gaussian likelihood with the THREESIX noise prior and a noise of at least 1e-4."""
    return GaussianLikelihood(
        noise_prior=THREESIX_NOISE_PRIOR(),
        noise_constraint=GreaterThan(lower_bound=1e-4),
    )


def MBO_LIKELIHOOD() -> GaussianLikelihood:
    """Gaussian likelihood with the MBO noise prior and a noise of at least 1e-4."""
    return GaussianLikelihood(
        noise_prior=MBO_NOISE_PRIOR(),
        noise_constraint=GreaterThan(lower_bound=1e-4),
    )


def HVARFNER_LIKELIHOOD() -> GaussianLikelihood:
    """Gaussian likelihood with the HVARFNER noise prior, BoTorch's default.

    The prior is LogNormal(-4, 1); the noise is at least 1e-4 and its fit starts at
    exp(-5), the prior's mode.
    """
    return GaussianLikelihood(
        noise_prior=HVARFNER_NOISE_PRIOR(),
        noise_constraint=GreaterThan(lower_bound=1e-4, initial_value=math.exp(-5.0)),
    )
