from pydantic import PositiveFloat

from bofire.data_models.base import BaseModel


class PredictedValue(BaseModel):
    """Container holding information regarding individual predictions.

    Used to comunicate with backend services.

    Attributes:
        predictedValue (float): The predicted value.
        standardDeviation (float): The standard deviation associated with the prediction.
            Has to be greater/equal than zero.
    """

    predictedValue: float
    standardDeviation: PositiveFloat
    # Annotated[List[str], Field(ge=0)]  # type: ignore
