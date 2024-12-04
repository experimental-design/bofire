from typing import Literal

from pydantic import PositiveFloat

from bofire.data_models.priors.gamma import GammaPrior
from bofire.data_models.priors.prior import Prior


class LKJPrior(Prior):
    """LKJ prior over correlation matrices. Allows to specify the shape of the prior.

    Attributes:
        n(int): number of dimensions of the correlation matrix
        eta(PositiveFloat): shape parameter of the LKJ distribution
        sd_prior(Prior): prior over the standard deviations of the correlation matrix

    """

    type: Literal["LKJPrior"] = "LKJPrior"
    shape: PositiveFloat
    sd_prior: GammaPrior
    n_tasks: int = 2
