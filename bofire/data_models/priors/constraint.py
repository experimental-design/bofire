from typing import Any, Literal, Optional

from pydantic import Field, PositiveFloat, model_validator

from bofire.data_models.base import BaseModel


class PriorConstraint(BaseModel):
    """Abstract Prior Constraint class."""

    type: Any
    initial_value: Optional[PositiveFloat] = Field(
        default=None,
        description="Value the constrained parameter starts from when the model is "
        "fitted. Must satisfy the constraint. If not provided, the parameter keeps "
        "whatever default the surrogate supplies.",
    )


class Positive(PriorConstraint):
    """Restricts a hyperparameter to values greater than zero.

    The loosest of the prior constraints: it rules out the non-positive half of the
    range and leaves everything else free.
    """

    type: Literal["Positive"] = "Positive"


class GreaterThan(PriorConstraint):
    """Restricts a hyperparameter to values at or above a lower bound."""

    type: Literal["GreaterThan"] = "GreaterThan"
    lower_bound: float = Field(
        description="Lower bound enforced on the constrained parameter.",
    )

    @model_validator(mode="after")
    def validate_initial_value(self):
        if self.initial_value is not None and self.initial_value < self.lower_bound:
            raise ValueError(
                "The initial value must be greater than or equal to the lower bound."
            )
        return self


class LessThan(PriorConstraint):
    """Restricts a hyperparameter to values at or below an upper bound."""

    type: Literal["LessThan"] = "LessThan"
    upper_bound: float = Field(
        description="Upper bound enforced on the constrained parameter.",
    )

    @model_validator(mode="after")
    def validate_initial_value(self):
        if self.initial_value is not None and self.initial_value > self.upper_bound:
            raise ValueError(
                "The initial value must be less than or equal to the upper bound."
            )
        return self
