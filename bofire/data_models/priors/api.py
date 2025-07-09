from functools import partial
from typing import Union

from bofire.data_models.priors.constraint import PriorConstraint
from bofire.data_models.priors.gamma import GammaPrior
from bofire.data_models.priors.lkj import LKJPrior
from bofire.data_models.priors.nontransformedinterval import NonTransformedInterval
from bofire.data_models.priors.normal import (
    DimensionalityScaledLogNormalPrior,
    LogNormalPrior,
    NormalPrior,
)
from bofire.data_models.priors.prior import Prior


AbstractPrior = Prior
AbstractPriorConstraint = PriorConstraint

AnyPrior = Union[
    GammaPrior,
    NormalPrior,
    LKJPrior,
    LogNormalPrior,
    DimensionalityScaledLogNormalPrior,
]

AnyPriorConstraint = NonTransformedInterval

# these are priors that are generally applicable
# and do not depend on problem specific extra parameters
AnyGeneralPrior = Union[GammaPrior, NormalPrior, LKJPrior, LogNormalPrior]

# default priors of interest
# botorch defaults
THREESIX_LENGTHSCALE_PRIOR = partial(GammaPrior, concentration=3.0, rate=6.0)
THREESIX_NOISE_PRIOR = partial(GammaPrior, concentration=1.1, rate=0.05)
THREESIX_SCALE_PRIOR = partial(GammaPrior, concentration=2.0, rate=0.15)

# mbo priors
# By default BoTorch places a highly informative prior on the kernel lengthscales,
# which easily leads to overfitting. Here we set a broader prior distribution for the
# lengthscale. The priors for the noise and signal variance are set more tightly.
MBO_LENGTHCALE_PRIOR = partial(GammaPrior, concentration=2.0, rate=0.2)
MBO_NOISE_PRIOR = partial(GammaPrior, concentration=2.0, rate=4.0)
MBO_OUTPUTSCALE_PRIOR = partial(GammaPrior, concentration=2.0, rate=4.0)

# prior for multitask kernel
LKJ_PRIOR = partial(
    LKJPrior,
    shape=2.0,
    sd_prior=GammaPrior(concentration=2.0, rate=0.15),
)

# prior for RobustSingleTaskGPSurrogate
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

# Hvarfner priors
HVARFNER_NOISE_PRIOR = partial(LogNormalPrior, loc=-4, scale=1)
HVARFNER_LENGTHSCALE_PRIOR = DimensionalityScaledLogNormalPrior

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
