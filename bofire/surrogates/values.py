from typing import Union

from pydantic import Field
from typing_extensions import Annotated

from bofire.data_models.base import BaseModel


class PredictedValue(BaseModel):
    """Container holding information regarding individual predictions.

    Used to comunicate with backend services.

    Attributes:
        predictedValue (float): The predicted value.
        standardDeviation (float): The standard deviation associated with the prediction.
            Has to be greater/equal than zero.
    """

    predictedValue: Union[float, str]
    standardDeviation: Annotated[float, Field(ge=0)]
