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
    CHEN_LENGTHSCALE_PRIOR,
    CHEN_OUTPUTSCALE_PRIOR,
    DIMENSIONALITY_SCALED_THREESIX_LENGTHSCALE_PRIOR,
    DimensionalityScaledGammaPrior,
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


@pytest.mark.parametrize(
    "concentration, concentration_scaling, rate, rate_power, d",
    [
        (8.0, 0.8, 2.0, 0.0, 6),  # CHEN-style lengthscale
        (3.0, 0.0, 10.0, -0.5, 9),  # threesix-style rate decay
    ],
)
def test_DimensionalityScaledGammaPrior_map(
    concentration,
    concentration_scaling,
    rate,
    rate_power,
    d,
):
    prior_data_model = DimensionalityScaledGammaPrior(
        concentration=concentration,
        concentration_scaling=concentration_scaling,
        rate=rate,
        rate_power=rate_power,
    )
    prior = priors.map(prior_data_model, d=d)
    assert isinstance(prior, gpytorch.priors.GammaPrior)
    assert prior.concentration == concentration + math.sqrt(d) * concentration_scaling
    assert prior.rate == rate * d**rate_power


@pytest.mark.parametrize("d", [1, 5, 20])
def test_chen_and_threesix_constants_map(d):
    # CHEN: lengthscale Gamma(2m, 2), outputscale Gamma(m, 1), m = 0.4*sqrt(d) + 4
    m = 0.4 * math.sqrt(d) + 4
    ls = priors.map(CHEN_LENGTHSCALE_PRIOR(), d=d)
    assert ls.concentration == pytest.approx(2 * m)
    assert ls.rate == pytest.approx(2.0)
    os = priors.map(CHEN_OUTPUTSCALE_PRIOR(), d=d)
    assert os.concentration == pytest.approx(m)
    assert os.rate == pytest.approx(1.0)

    # dimensionality-scaled threesix: concentration 3, rate base ~10.16 scaled by d**-0.5
    threesix = priors.map(DIMENSIONALITY_SCALED_THREESIX_LENGTHSCALE_PRIOR(), d=d)
    assert threesix.concentration == pytest.approx(3.0)
    assert threesix.rate == pytest.approx(2.0 / math.exp(math.sqrt(2) - 3) * d**-0.5)
