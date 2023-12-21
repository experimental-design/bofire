import gpytorch.priors
import pytest

import bofire.priors.api as priors
from bofire.data_models.priors.api import GammaPrior, NormalPrior


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
