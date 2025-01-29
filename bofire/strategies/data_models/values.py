from typing import Annotated

from pydantic import Field

from bofire.data_models.base import BaseModel


class InputValue(BaseModel):
    """Bofire input value.

    Attributes:
        value (Union[float, str, int]): The input value.

    """

    value: str


class OutputValue(BaseModel):
    """Bofire predicted output value.

    Attributes:
        predictedValue (Value): The predicted value.
        standardDeviation (float): Standard deviation, has to be zero or larger.
        objective (float): The objective value.

    """

    predictedValue: str
    standardDeviation: Annotated[float, Field(ge=0)]
    objective: float
