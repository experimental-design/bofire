import math
from typing import Literal

from pydantic import Field, PositiveFloat

from bofire.data_models.priors.prior import Prior


class NormalPrior(Prior):
    """Normal prior based on the normal distribution.

    Examples:
        >>> NormalPrior(loc=0.0, scale=1.0)
    """

    type: Literal["NormalPrior"] = "NormalPrior"
    loc: float = Field(description="Mean/center of the normal distribution.")
    scale: PositiveFloat = Field(description="Width of the normal distribution.")


class LogNormalPrior(Prior):
    """Log-normal prior based on the log-normal distribution.

    Use instead of `NormalPrior` for a parameter that must stay positive, such as a
    lengthscale.

    Examples:
        >>> LogNormalPrior(loc=0.0, scale=1.0)
    """

    type: Literal["LogNormalPrior"] = "LogNormalPrior"
    loc: float = Field(
        description="Mean/center of the log-normal distribution, on the log scale.",
    )
    scale: float = Field(
        description="Width of the log-normal distribution, on the log scale.",
    )


class DimensionalityScaledLogNormalPrior(Prior):
    """This prior is a log-normal prior where loc and scale are scaled by the dimensionaly of the problem.
    It was introduced by Hvarfner et al. in their paper https://arxiv.org/abs/2402.02229. More can be read in
    this excellent blogpost: https://www.miguelgondu.com/blogposts/2024-03-16/when-does-vanilla-gpr-fail/

    The effective log-normal distribution used at mapping time, given the
    dimensionality ``d``, is::

        loc_eff   = loc + log(d) * loc_scaling
        scale_eff = sqrt(scale**2 + log(d) * scale_scaling)

    Use instead of `LogNormalPrior` when the same prior should be reusable across
    problems of differing dimensionality.
    """

    type: Literal["DimensionalityScaledLogNormalPrior"] = (
        "DimensionalityScaledLogNormalPrior"
    )
    loc: PositiveFloat = Field(
        default=math.sqrt(2),
        description="Base mean/center of the log-normal distribution, before the "
        "dimensionality-dependent shift is added.",
    )
    loc_scaling: PositiveFloat = Field(
        default=0.5,
        description="Factor multiplying ``log(d)`` that is added to the base loc.",
    )
    scale: PositiveFloat = Field(
        default=math.sqrt(3),
        description="Base width of the log-normal distribution, before the "
        "dimensionality-dependent term is added.",
    )
    scale_scaling: float = Field(
        default=0.0,
        description="Factor multiplying ``log(d)`` that is added to the squared base "
        "scale.",
    )
