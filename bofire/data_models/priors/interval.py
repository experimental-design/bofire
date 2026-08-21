from typing import Literal

from pydantic import Field, PositiveFloat, model_validator

from bofire.data_models.priors.constraint import PriorConstraint


class Interval(PriorConstraint):
    """Restricts a hyperparameter to a closed interval."""

    type: Literal["Interval"] = "Interval"
    lower_bound: PositiveFloat = Field(description="Lower bound of the interval.")
    upper_bound: PositiveFloat = Field(description="Upper bound of the interval.")

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
