import gpytorch
from botorch.models.utils.gpytorch_modules import (
    get_gaussian_likelihood_with_lognormal_prior,
)

import bofire.likelihoods.api as likelihoods
from bofire.data_models.likelihoods.api import GaussianLikelihood
from bofire.data_models.priors.api import THREESIX_NOISE_PRIOR, GreaterThan


def _describe(likelihood: gpytorch.likelihoods.GaussianLikelihood) -> tuple:
    prior = likelihood.noise_covar.noise_prior
    constraint = likelihood.noise_covar.raw_noise_constraint
    return (
        type(prior).__name__,
        float(prior.loc),
        float(prior.scale),
        type(constraint).__name__,
        float(constraint.lower_bound),
        constraint._transform,
        round(float(constraint.initial_value), 12),
    )


def test_default_gaussian_likelihood_is_botorchs():
    """The default is exactly BoTorch's log-normal noise likelihood."""
    ours = likelihoods.map(GaussianLikelihood(), d=3)

    assert _describe(ours) == _describe(get_gaussian_likelihood_with_lognormal_prior())


def test_gaussian_likelihood_maps_prior_and_constraint():
    ours = likelihoods.map(
        GaussianLikelihood(
            noise_prior=THREESIX_NOISE_PRIOR(),
            noise_constraint=GreaterThan(lower_bound=0.5),
        ),
        d=3,
    )

    assert isinstance(ours.noise_covar.noise_prior, gpytorch.priors.GammaPrior)
    assert float(ours.noise_covar.raw_noise_constraint.lower_bound) == 0.5


def test_gaussian_likelihood_without_constraint():
    """Without a constraint, gpytorch's own default lower bound of 1e-4 applies."""
    ours = likelihoods.map(GaussianLikelihood(noise_constraint=None), d=3)

    constraint = ours.noise_covar.raw_noise_constraint
    assert isinstance(constraint, gpytorch.constraints.GreaterThan)
    assert abs(float(constraint.lower_bound) - 1e-4) < 1e-9
