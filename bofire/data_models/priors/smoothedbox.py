from typing import Literal

from pydantic import Field, PositiveFloat, model_validator

from bofire.data_models.priors.prior import Prior


class SmoothedBoxPrior(Prior):
    """A smoothed approximation of a uniform prior.

    .. math::

    \begin{equation*}
        B = {x: a_i <= x_i <= b_i}
        d(x, B) = min_{x' in B} |x - x'|
        pdf(x) \\sim exp(- d(x, B)**2 / sqrt(2 * sigma^2))
    \\end{equation*}

    Use when a hyperparameter should be treated as roughly uniform over a range, in
    contrast to the peaked `GammaPrior` or `NormalPrior`.

    Examples:
        >>> SmoothedBoxPrior(lower_bound=0.1, upper_bound=10.0)
    """

    type: Literal["SmoothedBoxPrior"] = "SmoothedBoxPrior"
    lower_bound: float = Field(
        description="Lower bound of the approximated uniform prior.",
    )
    upper_bound: float = Field(
        description="Upper bound of the approximated uniform prior.",
    )
    sigma: PositiveFloat = Field(
        default=0.01,
        description="Width of the smooth decay outside the bounds. Smaller values "
        "approximate a hard uniform prior more closely.",
    )

    @model_validator(mode="after")
    def validate_bounds(self):
        if self.lower_bound >= self.upper_bound:
            raise ValueError(
                "The lower bound must be less than the upper bound for an interval."
            )
        return self
