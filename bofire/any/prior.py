from typing import Union

from bofire.models.gps import priors

AnyPrior = Union[
    priors.GammaPrior,
    priors.NormalPrior,
]
