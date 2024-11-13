from typing import Annotated, Union

from pydantic import Field

from bofire.data_models.base import BaseModel


class PredictedValue(BaseModel):
    """Container holding information regarding individual predictions.

    Used to communicate with backend services.

    Attributes:
        predictedValue (float): The predicted value.
        standardDeviation (float): The standard deviation associated with the prediction.
            Has to be greater/equal than zero.

    """

    predictedValue: Union[float, str]
    standardDeviation: Annotated[float, Field(ge=0)]
