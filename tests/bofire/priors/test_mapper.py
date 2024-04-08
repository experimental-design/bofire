import gpytorch.priors
import pytest

import bofire.priors.api as priors
from bofire.data_models.priors.api import GammaPrior, LKJPrior, NormalPrior


@pytest.mark.parametrize(
    "prior, expected_prior",
    [
        (GammaPrior(concentration=2.0, rate=0.2), gpytorch.priors.GammaPrior),
        (NormalPrior(loc=0, scale=0.5), gpytorch.priors.NormalPrior),
    ],
)
def test_map(prior, expected_prior):
    gprior = priors.map(prior)
    assert isinstance(gprior, expected_prior)
    for key, value in prior.dict().items():
        if key == "type":
            continue
        assert value == getattr(gprior, key)


def test_lkj_map():
    prior = LKJPrior(
        n_tasks=3, shape=0.4, sd_prior=GammaPrior(concentration=2.0, rate=0.2)
    )
    expected_prior = gpytorch.priors.LKJPrior

    gprior = priors.map(prior)
    assert isinstance(gprior, expected_prior)
    assert prior.n_tasks == gprior.correlation_prior.n
    assert prior.shape == gprior.correlation_prior.concentration
    assert isinstance(gprior.sd_prior, gpytorch.priors.GammaPrior)
    assert prior.sd_prior.concentration == gprior.sd_prior.concentration
    assert prior.sd_prior.rate == gprior.sd_prior.rate
