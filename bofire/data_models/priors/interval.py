from typing import Literal

from pydantic import Field, model_validator

from bofire.data_models.priors.constraint import PriorConstraint


class Interval(PriorConstraint):
    """Restricts a hyperparameter to a closed interval."""

    type: Literal["Interval"] = "Interval"
    lower_bound: float = Field(description="Lower bound of the interval.")
    upper_bound: float = Field(description="Upper bound of the interval.")

    @model_validator(mode="after")
    def validate_bounds(self):
        if self.lower_bound >= self.upper_bound:
            raise ValueError(
                "The lower bound must be less than the upper bound for an interval."
            )
        if self.initial_value is not None and (
            self.initial_value < self.lower_bound
            or self.initial_value > self.upper_bound
        ):
            raise ValueError(
                "The initial value must be within the bounds of the interval.",
            )
        return self


class NonTransformedInterval(Interval):
    """NonTransformedInterval.

    Modification of the GPyTorch interval class that does not apply transformations.

    See: https://botorch.readthedocs.io/en/stable/_modules/botorch/utils/constraints.html#NonTransformedInterval
    """

    type: Literal["NonTransformedInterval"] = "NonTransformedInterval"


class LogTransformedInterval(Interval):
    """LogTransformedInterval.

    Modification of the GPyTorch interval class for numerical stability.

    See: https://botorch.readthedocs.io/en/stable/_modules/botorch/utils/constraints.html#LogTransformedInterval
    """

    type: Literal["LogTransformedInterval"] = "LogTransformedInterval"

    @model_validator(mode="after")
    def validate_positive_lower_bound(self):
        """The interval is optimized on a log scale, which needs positive bounds."""
        if self.lower_bound <= 0:
            raise ValueError(
                "The lower bound of a log-transformed interval must be positive."
            )
        return self
