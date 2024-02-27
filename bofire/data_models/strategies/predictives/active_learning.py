from typing import Literal, Optional, Type

from pydantic import Field, field_validator

from bofire.data_models.acquisition_functions.api import (
    AnyActiveLearningAcquisitionFunction,
    qNegIntPosVar,
)
from bofire.data_models.features.api import CategoricalOutput, Feature
from bofire.data_models.objectives.api import ConstrainedObjective, Objective
from bofire.data_models.strategies.predictives.botorch import BotorchStrategy


class ActiveLearningStrategy(BotorchStrategy):
    type: Literal["ActiveLearningStrategy"] = "ActiveLearningStrategy"
    acquisition_function: AnyActiveLearningAcquisitionFunction = Field(
        default_factory=lambda: qNegIntPosVar()
    )

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
    
    @field_validator("domain")
    @classmethod
    def validate_is_singleobjective(cls, v, values):
        if len(v.outputs) == 1:
            return v
        if (
            len(v.outputs.get_by_objective(excludes=ConstrainedObjective))
            - len(v.outputs.get_by_objective(includes=None, excludes=Objective))
        ) > 1:
            raise ValueError(
                "Active learning strategy can only deal with one objective."
            )
        return v
