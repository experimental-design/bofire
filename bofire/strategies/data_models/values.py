from typing import Annotated, Union

from pydantic import Field

from bofire.data_models.base import BaseModel

Value = Union[float, str, int]


class InputValue(BaseModel):
    """Bofire input value.

    Attributes:
        value (Union[float, str, int]): The input value.
    """

    value: Value


class OutputValue(BaseModel):
    """Bofire predicted output value.

    Attributes:
        predictedValue (Value): The predicted value.
        standardDeviation (float): Standard deviation, has to be zero or larger.
        objective (float): The objective value.
    """

    predictedValue: Value
    standardDeviation: Annotated[float, Field(ge=0)]
    objective: float
