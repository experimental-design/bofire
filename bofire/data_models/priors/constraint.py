from typing import Any, Literal

from bofire.data_models.base import BaseModel


class PriorConstraint(BaseModel):
    """Abstract Prior Constraint class."""

    type: Any


class Positive(PriorConstraint):
    """Class for constraints that enforce a prior to be positive.

    Attributes:
        type (Literal): A string literal to identify the class type.
    """

    type: Literal["Positive"] = "Positive"


class GreaterThan(PriorConstraint):
    """Class for constraints that enforce a prior to be greater than a specified value.

    Attributes:
        type (Literal): A string literal to identify the class type.
    """

    type: Literal["GreaterThan"] = "GreaterThan"
    lower_bound: float


class LessThan(PriorConstraint):
    """Class for constraints that enforce a prior to be less than a specified value.

    Attributes:
        type (Literal): A string literal to identify the class type.
    """

    type: Literal["LessThan"] = "LessThan"
    upper_bound: float
