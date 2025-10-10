import math

import gpytorch.priors
import numpy as np
import pytest
from botorch.utils.constraints import (
    LogTransformedInterval as BotorchLogTransformedInterval,
)
from botorch.utils.constraints import (
    NonTransformedInterval as BotorchNonTransformedInterval,
)

import bofire.priors.api as priors
from bofire.data_models.priors.api import (
    DimensionalityScaledLogNormalPrior,
    GammaPrior,
    GreaterThan,
    LessThan,
    LKJPrior,
    LogNormalPrior,
    LogTransformedInterval,
    NonTransformedInterval,
    NormalPrior,
    Positive,
)


@pytest.mark.parametrize(
    "prior, expected_prior",
    [
        (GammaPrior(concentration=2.0, rate=0.2), gpytorch.priors.GammaPrior),
        (NormalPrior(loc=0, scale=0.5), gpytorch.priors.NormalPrior),
        (LogNormalPrior(loc=0, scale=0.5), gpytorch.priors.LogNormalPrior),
        (
            NonTransformedInterval(lower_bound=0.1, upper_bound=1.0, initial_value=0.5),
            BotorchNonTransformedInterval,
        ),
        (
            LogTransformedInterval(lower_bound=0.1, upper_bound=1.0, initial_value=0.5),
            BotorchLogTransformedInterval,
        ),
        (Positive(), gpytorch.constraints.Positive),
        (GreaterThan(lower_bound=0), gpytorch.constraints.GreaterThan),
        (LessThan(upper_bound=0), gpytorch.constraints.LessThan),
    ],
)
def test_map(prior, expected_prior):
    gprior = priors.map(prior)
    assert isinstance(gprior, expected_prior)
    for key, value in prior.model_dump().items():
        if key == "type":
            continue
        if not isinstance(prior, LogTransformedInterval):
            assert value == getattr(gprior, key)


def test_lkj_map():
    prior = LKJPrior(
        n_tasks=3,
        shape=0.4,
        sd_prior=GammaPrior(concentration=2.0, rate=0.2),
    )
    expected_prior = gpytorch.priors.LKJPrior

    gprior = priors.map(prior)
    assert isinstance(gprior, expected_prior)
    assert prior.n_tasks == gprior.correlation_prior.n
    assert prior.shape == gprior.correlation_prior.concentration
    assert isinstance(gprior.sd_prior, gpytorch.priors.GammaPrior)
    assert prior.sd_prior.concentration == gprior.sd_prior.concentration
    assert prior.sd_prior.rate == gprior.sd_prior.rate


@pytest.mark.parametrize(
    "loc, loc_scaling, scale, scale_scaling, d",
    [
        (np.sqrt(2), 0.5, np.sqrt(3), 0.0, 6),
    ],
)
def test_DimensionalityScaledLogNormalPrior_map(
    loc,
    loc_scaling,
    scale,
    scale_scaling,
    d,
):
    prior_data_model = DimensionalityScaledLogNormalPrior(
        loc=loc,
        loc_scaling=loc_scaling,
        scale=scale,
        scale_scaling=scale_scaling,
    )
    prior = priors.map(prior_data_model, d=d)
    assert isinstance(prior, gpytorch.priors.LogNormalPrior)
    assert prior.loc == loc + math.log(d) * loc_scaling
    assert prior.scale == (scale**2 + math.log(d) * scale_scaling) ** 0.5
