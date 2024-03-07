from typing import Dict, List, Literal, Optional, Type

from pydantic import Field, field_validator

from bofire.data_models.acquisition_functions.api import (
    AnyActiveLearningAcquisitionFunction,
    qNegIntPosVar,
)
from bofire.data_models.domain.api import Domain
from bofire.data_models.features.api import CategoricalOutput, Feature
from bofire.data_models.objectives.api import Objective
from bofire.data_models.strategies.predictives.botorch import BotorchStrategy


class ActiveLearningStrategy(BotorchStrategy):
    type: Literal["ActiveLearningStrategy"] = "ActiveLearningStrategy"
    acquisition_function: AnyActiveLearningAcquisitionFunction = Field(
        default_factory=lambda: qNegIntPosVar()
    )
    weights: Optional[Dict[str, float]] = Field(default_factory=lambda: None)

    @classmethod
    def is_feature_implemented(cls, my_type: Type[Feature]) -> bool:
        """Method to check if a specific feature type is implemented for the strategy

        Args:
            my_type (Type[Feature]): Feature class

        Returns:
            bool: True if the feature type is valid for the strategy chosen, False otherwise
        """
        if my_type not in [CategoricalOutput]:
            return True
        return False

    @classmethod
    def is_objective_implemented(cls, my_type: Type[Objective]) -> bool:
        """Method to check if a objective type is implemented for the strategy

        Args:
            my_type (Type[Objective]): Objective class

        Returns:
            bool: True if the objective type is valid for the strategy chosen, False otherwise
        """
        return True
