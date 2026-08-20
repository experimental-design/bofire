from typing import Literal, Optional

from pydantic import Field, PositiveFloat, model_validator

from bofire.data_models.priors.constraint import PriorConstraint


class Interval(PriorConstraint):
    """Interval constraint on a GP hyperparameter.

    It is used to define interval constraints on GP hyperparameters.

    Bounds the parameter from both sides, unlike `GreaterThan`, `LessThan` and
    `Positive`.
    """

    type: Literal["Interval"] = "Interval"
    lower_bound: PositiveFloat = Field(description="Lower bound of the interval.")
    upper_bound: PositiveFloat = Field(description="Upper bound of the interval.")
    initial_value: Optional[PositiveFloat] = Field(
        default=None,
        description="Optional warm-start value used when registering the constraint "
        "on a gpytorch parameter. Must lie within the interval. If not provided, "
        "gpytorch leaves the raw parameter at its default.",
    )

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
