from typing import Any, Literal, Optional

from pydantic import Field, PositiveFloat, model_validator

from bofire.data_models.base import BaseModel


class PriorConstraint(BaseModel):
    """Abstract Prior Constraint class."""

    type: Any


class Positive(PriorConstraint):
    """Class for constraints that enforce a prior to be positive.

    Use when only the sign matters; use `GreaterThan` or `Interval` to bound the
    parameter more tightly.
    """

    type: Literal["Positive"] = "Positive"
    initial_value: Optional[PositiveFloat] = Field(
        default=None,
        description="Optional warm-start value used when registering the constraint "
        "on a gpytorch parameter. If not provided, the consuming code may supply a "
        "runtime default, e.g. ``noise_prior.mode`` in the GP surrogates.",
    )


class GreaterThan(PriorConstraint):
    """Class for constraints that enforce a prior to be greater than a specified value.

    Use `Interval` instead if the parameter needs an upper bound as well.
    """

    type: Literal["GreaterThan"] = "GreaterThan"
    lower_bound: float = Field(
        description="Lower bound enforced on the constrained parameter.",
    )
    initial_value: Optional[PositiveFloat] = Field(
        default=None,
        description="Optional warm-start value used when registering the constraint "
        "on a gpytorch parameter. Must be greater than or equal to `lower_bound`. If "
        "not provided, the consuming code may supply a runtime default, e.g. "
        "``noise_prior.mode`` in the GP surrogates.",
    )

    @model_validator(mode="after")
    def validate_initial_value(self):
        if self.initial_value is not None and self.initial_value < self.lower_bound:
            raise ValueError(
                "The initial value must be greater than or equal to the lower bound."
            )
        return self


class LessThan(PriorConstraint):
    """Class for constraints that enforce a prior to be less than a specified value.

    Use `Interval` instead if the parameter needs a lower bound as well.
    """

    type: Literal["LessThan"] = "LessThan"
    upper_bound: float = Field(
        description="Upper bound enforced on the constrained parameter.",
    )
    initial_value: Optional[PositiveFloat] = Field(
        default=None,
        description="Optional warm-start value used when registering the constraint "
        "on a gpytorch parameter. Must be less than or equal to `upper_bound`. If not "
        "provided, the consuming code may supply a runtime default, e.g. "
        "``noise_prior.mode`` in the GP surrogates.",
    )

    @model_validator(mode="after")
    def validate_initial_value(self):
        if self.initial_value is not None and self.initial_value > self.upper_bound:
            raise ValueError(
                "The initial value must be less than or equal to the upper bound."
            )
        return self
