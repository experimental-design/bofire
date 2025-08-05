from typing import Any, Literal

from pydantic import PositiveFloat, model_validator

from bofire.data_models.priors.constraint import PriorConstraint


class Interval(PriorConstraint):
    """Abstract Interval class.

    It is used to define interval constraints on GP hyperparameters.

    Attributes:
        lower_bound: The lower bound of the interval.
        upper_bound: The upper bound of the interval.
        initial_value: The initial value within the interval.
    """

    type: Any
    lower_bound: PositiveFloat
    upper_bound: PositiveFloat
    initial_value: PositiveFloat

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


class NonTransformedInterval(Interval):
    """NonTransformedInterval.

    Modification of the GPyTorch interval class that does not apply transformations.

    See: https://botorch.readthedocs.io/en/stable/_modules/botorch/utils/constraints.html#NonTransformedInterval

    Attributes:
        lower_bound: The lower bound of the interval.
        upper_bound: The upper bound of the interval.
        initial_value: The initial value within the interval.

    """

    type: Literal["NonTransformedInterval"] = "NonTransformedInterval"


class LogTransformedInterval(Interval):
    """LogTransformedInterval.

    Modification of the GPyTorch interval class for numerical stability.

    See: https://botorch.readthedocs.io/en/stable/_modules/botorch/utils/constraints.html#LogTransformedInterval

    Attributes:
        lower_bound: The lower bound of the interval.
        upper_bound: The upper bound of the interval.
        initial_value: The initial value within the interval.
    """

    type: Literal["LogTransformedInterval"] = "LogTransformedInterval"
