from typing import Literal

from pydantic import Field

from bofire.data_models.acquisition_functions.api import (
    AnyActiveLearningAcquisitionFunction,
    qNegIntPosVar,
)
from bofire.data_models.strategies.predictives.botorch import BotorchStrategy


class ActiveLearningStrategy(BotorchStrategy):
    type: Literal["ActiveLearningStrategy"] = "ActiveLearningStrategy"
    acquisition_function: AnyActiveLearningAcquisitionFunction = Field(
        default_factory=lambda: qNegIntPosVar()
    )
