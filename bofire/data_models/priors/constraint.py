from typing import Any, Literal, Optional

from pydantic import PositiveFloat, model_validator

from bofire.data_models.base import BaseModel


class PriorConstraint(BaseModel):
    """Abstract Prior Constraint class."""

    type: Any


class Positive(PriorConstraint):
    """Class for constraints that enforce a prior to be positive.

    Attributes:
        initial_value: Optional warm-start value used when registering the
            constraint on a gpytorch parameter. If ``None``, the consuming code
            may supply a runtime default (e.g. ``noise_prior.mode`` in the GP
            surrogates).
    """

    type: Literal["Positive"] = "Positive"
    initial_value: Optional[PositiveFloat] = None


class GreaterThan(PriorConstraint):
    """Class for constraints that enforce a prior to be greater than a specified value.

    Attributes:
        lower_bound: The lower bound enforced on the constrained parameter.
        initial_value: Optional warm-start value used when registering the
            constraint on a gpytorch parameter. Must be ``>= lower_bound`` if
            set. If ``None``, the consuming code may supply a runtime default
            (e.g. ``noise_prior.mode`` in the GP surrogates).
    """

    type: Literal["GreaterThan"] = "GreaterThan"
    lower_bound: float
    initial_value: Optional[PositiveFloat] = None

    @model_validator(mode="after")
    def validate_initial_value(self):
        if self.initial_value is not None and self.initial_value < self.lower_bound:
            raise ValueError(
                "The initial value must be greater than or equal to the lower bound."
            )
        return self


class LessThan(PriorConstraint):
    """Class for constraints that enforce a prior to be less than a specified value.

    Attributes:
        upper_bound: The upper bound enforced on the constrained parameter.
        initial_value: Optional warm-start value used when registering the
            constraint on a gpytorch parameter. Must be ``<= upper_bound`` if
            set. If ``None``, the consuming code may supply a runtime default
            (e.g. ``noise_prior.mode`` in the GP surrogates).
    """

    type: Literal["LessThan"] = "LessThan"
    upper_bound: float
    initial_value: Optional[PositiveFloat] = None

    @model_validator(mode="after")
    def validate_initial_value(self):
        if self.initial_value is not None and self.initial_value > self.upper_bound:
            raise ValueError(
                "The initial value must be less than or equal to the upper bound."
            )
        return self
