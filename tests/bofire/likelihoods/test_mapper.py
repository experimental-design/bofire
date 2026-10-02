import gpytorch
import pytest
import torch
from botorch.models.likelihoods.pairwise import (
    PairwiseLogitLikelihood as BotorchPairwiseLogitLikelihood,
)
from botorch.models.likelihoods.pairwise import (
    PairwiseProbitLikelihood as BotorchPairwiseProbitLikelihood,
)
from botorch.models.utils.gpytorch_modules import (
    get_gaussian_likelihood_with_lognormal_prior,
)
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods.hadamard_gaussian_likelihood import HadamardGaussianLikelihood

import bofire.likelihoods.api as likelihoods
from bofire.data_models.likelihoods.api import (
    HVARFNER_LIKELIHOOD,
    GaussianLikelihood,
    PairwiseLogitLikelihood,
    PairwiseProbitLikelihood,
    TaskwiseLikelihood,
)
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


def test_hvarfner_likelihood_is_botorchs():
    """The preset is exactly BoTorch's log-normal noise likelihood."""
    ours = likelihoods.map(HVARFNER_LIKELIHOOD(), d=3)

    assert _describe(ours) == _describe(get_gaussian_likelihood_with_lognormal_prior())


def test_default_gaussian_likelihood_has_no_prior():
    ours = likelihoods.map(GaussianLikelihood(), d=3)

    assert list(ours.named_priors()) == []
    assert abs(float(ours.noise_covar.raw_noise_constraint.lower_bound) - 1e-4) < 1e-9


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


@pytest.mark.parametrize("task_feature_index", [0, 1])
def test_taskwise_likelihood_selects_noise_for_each_observation(task_feature_index):
    ours = likelihoods.map(
        TaskwiseLikelihood(likelihood=HVARFNER_LIKELIHOOD()),
        d=1,
        num_tasks=2,
        task_feature_index=task_feature_index,
    )
    assert isinstance(ours, HadamardGaussianLikelihood)
    assert _describe(ours) == _describe(get_gaussian_likelihood_with_lognormal_prior())
    torch.testing.assert_close(ours.noise, torch.exp(torch.tensor(-5.0)).expand(2))

    ours.noise = torch.tensor([0.1, 0.7])
    inputs = torch.zeros(4, 2)
    inputs[:, task_feature_index] = torch.tensor([1, 0, 1, 0])
    latent = MultivariateNormal(torch.zeros(4), torch.eye(4))
    observed = ours(latent, inputs)
    torch.testing.assert_close(
        observed.variance - latent.variance, torch.tensor([0.7, 0.1, 0.7, 0.1])
    )


@pytest.mark.parametrize(
    "data_model, expected_cls",
    [
        (PairwiseProbitLikelihood(), BotorchPairwiseProbitLikelihood),
        (PairwiseLogitLikelihood(), BotorchPairwiseLogitLikelihood),
    ],
)
def test_pairwise_likelihood(data_model, expected_cls):
    assert isinstance(likelihoods.map(data_model, d=3), expected_cls)
