from typing import Literal

from pydantic import Float, PositiveFloat, model_validator

from bofire.data_models.priors.prior import Prior


class SmoothedBoxPrior(Prior):
    """A smoothed approximation of a uniform prior.

    .. math::

    \begin{equation*}
        B = {x: a_i <= x_i <= b_i}
        d(x, B) = min_{x' in B} |x - x'|
        pdf(x) \\sim exp(- d(x, B)**2 / sqrt(2 * sigma^2))
    \\end{equation*}

    Attributes:
        lower_bound: lower bound of the uniform prior
        upper_bound: upper bound of the uniform prior
        sigma: related to pdf(x)

    """

    type: Literal["SmoothedBoxPrior"] = "SmoothedBoxPrior"
    lower_bound: Float
    upper_bound: Float
    sigma: PositiveFloat = 0.01

    @model_validator(mode="after")
    def validate_bounds(self):
        if self.lower_bound >= self.upper_bound:
            raise ValueError(
                "The lower bound must be less than the upper bound for an interval."
            )
        if (
            self.initial_value < self.lower_bound
            or self.initial_value > self.upper_bound
        ):
            raise ValueError(
                "The initial value must be within the bounds of the interval.",
            )
        return self
