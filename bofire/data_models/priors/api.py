import math
from functools import partial

from bofire.data_models.priors._register import (
    register_prior,
    register_prior_constraint,
)
from bofire.data_models.priors.constraint import (
    GreaterThan,
    LessThan,
    Positive,
    PriorConstraint,
)
from bofire.data_models.priors.gamma import DimensionalityScaledGammaPrior, GammaPrior
from bofire.data_models.priors.interval import (
    Interval,
    LogTransformedInterval,
    NonTransformedInterval,
)
from bofire.data_models.priors.lkj import LKJPrior
from bofire.data_models.priors.normal import (
    DimensionalityScaledLogNormalPrior,
    LogNormalPrior,
    NormalPrior,
)
from bofire.data_models.priors.prior import Prior
from bofire.data_models.priors.smoothedbox import SmoothedBoxPrior
from bofire.data_models.unions import tagged_union


_PRIOR_TYPES: list[type[Prior]] = [
    GammaPrior,
    DimensionalityScaledGammaPrior,
    NormalPrior,
    LKJPrior,
    LogNormalPrior,
    DimensionalityScaledLogNormalPrior,
    SmoothedBoxPrior,
]

AnyPrior = tagged_union(*_PRIOR_TYPES)

_PRIOR_CONSTRAINT_TYPES: list[type] = [
    Interval,
    NonTransformedInterval,
    LogTransformedInterval,
    Positive,
    GreaterThan,
    LessThan,
]

AnyPriorConstraint = tagged_union(*_PRIOR_CONSTRAINT_TYPES)


# default priors of interest
# botorch defaults
THREESIX_LENGTHSCALE_PRIOR = partial(GammaPrior, concentration=3.0, rate=6.0)
THREESIX_NOISE_PRIOR = partial(GammaPrior, concentration=1.1, rate=0.05)
THREESIX_SCALE_PRIOR = partial(GammaPrior, concentration=2.0, rate=0.15)

# mbo priors
# By default BoTorch places a highly informative prior on the kernel lengthscales,
# which easily leads to overfitting. Here we set a broader prior distribution for the
# lengthscale. The priors for the noise and signal variance are set more tightly.
MBO_LENGTHSCALE_PRIOR = partial(GammaPrior, concentration=2.0, rate=0.2)
MBO_NOISE_PRIOR = partial(GammaPrior, concentration=2.0, rate=4.0)
MBO_OUTPUTSCALE_PRIOR = partial(GammaPrior, concentration=2.0, rate=4.0)

# prior for multitask kernel
LKJ_PRIOR = partial(
    LKJPrior,
    shape=2.0,
    sd_prior=GammaPrior(concentration=2.0, rate=0.15),
)

# priors for RobustSingleTaskGPSurrogate
ROBUSTGP_LENGTHSCALE_CONSTRAINT = partial(
    NonTransformedInterval,
    lower_bound=0.05,
    upper_bound=float("inf"),
    initial_value=0.2,
)

ROBUSTGP_OUTPUTSCALE_CONSTRAINT = partial(
    NonTransformedInterval,
    lower_bound=0.01,
    upper_bound=10.0,
    initial_value=0.1,
)


# Priors for PairwiseGPSurrogate based on botorch defaults
PAIRWISEGP_LENGTHSCALE_PRIOR = partial(
    GammaPrior,
    concentration=2.4,
    rate=2.7,
)

PAIRWISEGP_LENGTHSCALE_CONSTRAINT = partial(
    GreaterThan,
    lower_bound=1e-4,
    initial_value=0.5185,  # mode of the lengthscale GammaPrior(2.4, 2.7)
)

PAIRWISEGP_OUTPUTSCALE_PRIOR = partial(
    SmoothedBoxPrior,
    lower_bound=0.01,
    upper_bound=100,
    sigma=0.01,
)

PAIRWISEGP_OUTPUTSCALE_CONSTRAINT = partial(
    Interval,
    lower_bound=5e-3,
    upper_bound=200,
    initial_value=1,
)

# Hvarfner priors
HVARFNER_NOISE_PRIOR = partial(LogNormalPrior, loc=-4, scale=1)
HVARFNER_LENGTHSCALE_PRIOR = DimensionalityScaledLogNormalPrior

# CHEN priors
# Dimension-aware hyperpriors proposed by Chen, Fleck and Stuyver, "Leveraging
# Hidden-Space Representations Effectively in Bayesian Optimization for Experiment
# Design through Dimension-Aware Hyperpriors", ChemRxiv (2026),
# https://doi.org/10.26434/chemrxiv.10001986/v2 (the CHEN preset in BayBE).
# The lengthscale follows a Gamma(2m, 2) and the outputscale a Gamma(m, 1) with
# m = 0.4 * sqrt(d) + 4, i.e. the concentration grows with the square root of the
# problem dimensionality d.
CHEN_LENGTHSCALE_PRIOR = partial(
    DimensionalityScaledGammaPrior,
    concentration=8.0,  # 2 * 4
    concentration_scaling=0.8,  # 2 * 0.4
    rate=2.0,
    rate_power=0.0,
)
CHEN_OUTPUTSCALE_PRIOR = partial(
    DimensionalityScaledGammaPrior,
    concentration=4.0,
    concentration_scaling=0.4,
    rate=1.0,
    rate_power=0.0,
)

# Dimensionality-scaled threesix lengthscale prior (BayBE's new default for search
# spaces without molecular parameters): keep the threesix concentration of 3 and scale
# the rate by d ** -0.5 so the lengthscale mode grows with sqrt(d). The base rate is
# calibrated such that the mode matches the Hvarfner log-normal lengthscale prior.
DIMENSIONALITY_SCALED_THREESIX_LENGTHSCALE_PRIOR = partial(
    DimensionalityScaledGammaPrior,
    concentration=3.0,
    concentration_scaling=0.0,
    rate=2.0 / math.exp(math.sqrt(2) - 3),  # ~10.16
    rate_power=-0.5,
)

# EDBO priors:
# adapted from the EDBO paper https://github.com/b-shields/edbo/blob/master/edbo/bro.py#L664
# and code https://doi.org/10.1038/s41586-021-03213-y
# EDBO also define starting values for the hyperparameters, which are currently not supported
# in BoFire. We provide it here as a comment behind the prior for reference.
EDBO_MORDRED_LENGTHSCALE_PRIOR = partial(
    GammaPrior, concentration=2.0, rate=0.1
)  # starting value 10.0
EDBO_MORDRED_OUTPUT_SCALE_PRIOR = partial(
    GammaPrior, concentration=2.0, rate=0.1
)  # starting value 10.0
EDBO_MORDRED_NOISE_PRIOR = partial(
    GammaPrior, concentration=1.5, rate=0.1
)  # starting value 5.0

EDBO_DFT_LENGTHSCALE_PRIOR = partial(
    GammaPrior, concentration=2.0, rate=0.2
)  # starting value 5.0
EDBO_DFT_OUTPUT_SCALE_PRIOR = partial(
    GammaPrior, concentration=5.0, rate=0.5
)  # starting value 8.0
EDBO_DFT_NOISE_PRIOR = partial(
    GammaPrior, concentration=1.5, rate=0.1
)  # starting value 5.0

EDBO_OHE_LENGTHSCALE_PRIOR = partial(
    GammaPrior, concentration=3.0, rate=1.0
)  # starting value 2.0
EDBO_OHE_OUTPUT_SCALE_PRIOR = partial(
    GammaPrior, concentration=5.0, rate=0.2
)  # starting value 20.0
EDBO_OHE_NOISE_PRIOR = partial(
    GammaPrior, concentration=1.5, rate=0.1
)  # starting value 5.0
