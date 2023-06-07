from functools import partial
from typing import Union

from bofire.data_models.priors.gamma import GammaPrior
from bofire.data_models.priors.normal import NormalPrior
from bofire.data_models.priors.prior import Prior

AbstractPrior = Prior

AnyPrior = Union[
    GammaPrior,
    NormalPrior,
]

# default priors of interest
# botorch defaults
BOTORCH_LENGTHCALE_PRIOR = partial(GammaPrior, concentration=3.0, rate=6.0)
BOTORCH_NOISE_PRIOR = partial(GammaPrior, concentration=1.1, rate=0.05)
BOTORCH_SCALE_PRIOR = partial(GammaPrior, concentration=2.0, rate=0.15)

# mbo priors
# By default BoTorch places a highly informative prior on the kernel lengthscales,
# which easily leads to overfitting. Here we set a broader prior distribution for the
# lengthscale. The priors for the noise and signal variance are set more tightly.
MBO_LENGTHCALE_PRIOR = partial(GammaPrior, concentration=2.0, rate=0.2)
MBO_NOISE_PRIOR = partial(GammaPrior, concentration=2.0, rate=4.0)
MBO_OUTPUTSCALE_PRIOR = partial(GammaPrior, concentration=2.0, rate=4.0)
