from typing import Literal

import gpytorch
from pydantic import PositiveFloat

from bofire.data_models.priors.prior import Prior


class NormalPrior(Prior):
    """Normal prior based on the normal distribution

    Attributes:
        loc(float): mean/center of the normal distribution
        scale(PositiveFloat): width of the normal distribution
    """

    type: Literal["NormalPrior"] = "NormalPrior"
    loc: float
    scale: PositiveFloat

    def to_gpytorch(self) -> gpytorch.priors.NormalPrior:
        return gpytorch.priors.NormalPrior(loc=self.loc, scale=self.scale)
