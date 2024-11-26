from functools import partial
from typing import Union

from bofire.data_models.priors.gamma import GammaPrior
from bofire.data_models.priors.lkj import LKJPrior
from bofire.data_models.priors.normal import (
    DimensionalityScaledLogNormalPrior,
    LogNormalPrior,
    NormalPrior,
)
from bofire.data_models.priors.prior import Prior


AbstractPrior = Prior

AnyPrior = Union[
    GammaPrior,
    NormalPrior,
    LKJPrior,
    LogNormalPrior,
    DimensionalityScaledLogNormalPrior,
]

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

# Hvarfner priors
HVARFNER_NOISE_PRIOR = partial(LogNormalPrior, loc=-4, scale=1)
HVARFNER_LENGTHSCALE_PRIOR = DimensionalityScaledLogNormalPrior
