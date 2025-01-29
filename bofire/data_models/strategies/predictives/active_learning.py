from typing import Literal, Type

from pydantic import Field, model_validator

from bofire.data_models.acquisition_functions.api import (
    AnyActiveLearningAcquisitionFunction,
    qNegIntPosVar,
)
from bofire.data_models.features.api import CategoricalOutput, Feature
from bofire.data_models.objectives.api import Objective
from bofire.data_models.strategies.predictives.botorch import BotorchStrategy


class ActiveLearningStrategy(BotorchStrategy):
    """Datamodel for an ActiveLearningStrategy that focusses on pure exploration of the input space.
    This type of strategy chooses new candidate points in order to minimize the uncertainty.
    """

    type: Literal["ActiveLearningStrategy"] = "ActiveLearningStrategy"
    acquisition_function: AnyActiveLearningAcquisitionFunction = Field(
        default_factory=lambda: qNegIntPosVar(),
    )

    @model_validator(mode="after")
    def validate_acquisition_function(self):
        if isinstance(self.acquisition_function, qNegIntPosVar):
            if self.acquisition_function.weights is not None:
                if sorted(self.acquisition_function.weights.keys()) != sorted(
                    self.domain.outputs.get_keys(),
                ):
                    raise ValueError(
                        "The keys provided for the weights do not match the required keys of the output features.",
                    )
        return self

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
