import gpytorch
from botorch.models.map_saas import get_mean_module_with_normal_prior

import bofire.means.api as means
from bofire.data_models.means.api import ConstantMean
from bofire.data_models.priors.api import NormalPrior


def _describe(mean: gpytorch.means.ConstantMean) -> tuple:
    ((_, _, prior, *_),) = mean.named_priors()
    constraint = mean.raw_constant_constraint
    return (
        type(prior).__name__,
        float(prior.loc),
        float(prior.scale),
        type(constraint).__name__,
        float(constraint.lower_bound),
        float(constraint.upper_bound),
        constraint._transform,
        float(constraint.initial_value),
    )


def test_default_constant_mean_is_unconstrained():
    """The default matches what a BoTorch single-task GP builds when given no mean."""
    ours = means.map(ConstantMean(), d=3)

    assert isinstance(ours, gpytorch.means.ConstantMean)
    assert list(ours.named_priors()) == []


def test_constant_mean_with_prior_and_bounds_is_botorchs():
    """Normal(0, 1) bounded to [-10, 10] is the mean of BoTorch's MAP-SAAS models."""
    ours = means.map(
        ConstantMean(prior=NormalPrior(loc=0.0, scale=1.0), bounds=(-10.0, 10.0)),
        d=3,
    )

    assert _describe(ours) == _describe(get_mean_module_with_normal_prior())
