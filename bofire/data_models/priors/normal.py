from typing import Literal

import numpy as np
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


class LogNormalPrior(Prior):
    """Log-normal prior based on the log-normal distribution

    Attributes:
        loc(float): mean/center of the log-normal distribution
        scale(PositiveFloat): width of the log-normal distribution

    """

    type: Literal["LogNormalPrior"] = "LogNormalPrior"
    loc: float
    scale: float


class DimensionalityScaledLogNormalPrior(Prior):
    """This prior is a log-normal prior where loc and scale are scaled by the dimensionaly of the problem.
    It was introduced by Hvarfner et al. in their paper https://arxiv.org/abs/2402.02229. More can be read in
    this excellent blogpost: https://www.miguelgondu.com/blogposts/2024-03-16/when-does-vanilla-gpr-fail/
    """

    type: Literal["DimensionalityScaledLogNormalPrior"] = (
        "DimensionalityScaledLogNormalPrior"
    )
    loc: PositiveFloat = np.sqrt(2)
    loc_scaling: PositiveFloat = 0.5
    scale: PositiveFloat = np.sqrt(3)
    scale_scaling: float = 0.0
