import gpytorch

import bofire.means.api as means
from bofire.data_models.means.api import ConstantMean
from bofire.data_models.priors.api import GreaterThan, NormalPrior


def test_default_constant_mean_is_unconstrained():
    """The default matches what a BoTorch single-task GP builds when given no mean."""
    ours = means.map(ConstantMean(), d=3)

    assert isinstance(ours, gpytorch.means.ConstantMean)
    assert list(ours.named_priors()) == []


def test_constant_mean_maps_prior_and_constraint():
    ours = means.map(
        ConstantMean(
            prior=NormalPrior(loc=0.0, scale=1.0),
            constraint=GreaterThan(lower_bound=-10.0),
        ),
        d=3,
    )

    ((_, _, prior, *_),) = ours.named_priors()
    assert isinstance(prior, gpytorch.priors.NormalPrior)
    assert float(ours.raw_constant_constraint.lower_bound) == -10.0
