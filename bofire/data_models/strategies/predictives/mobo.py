from typing import Dict, Literal, Optional, Type

from pydantic import Field, model_validator

from bofire.data_models.acquisition_functions.api import (
    AnyMultiObjectiveAcquisitionFunction,
    qLogNEHVI,
)
from bofire.data_models.features.api import CategoricalOutput, Feature
from bofire.data_models.objectives.api import (
    CloseToTargetObjective,
    MaximizeObjective,
    MaximizeSigmoidObjective,
    MinimizeObjective,
    MinimizeSigmoidObjective,
    Objective,
    TargetObjective,
)
from bofire.data_models.strategies.predictives.multiobjective import (
    MultiobjectiveStrategy,
)


class MoboStrategy(MultiobjectiveStrategy):
    type: Literal["MoboStrategy"] = "MoboStrategy"
    ref_point: Optional[Dict[str, float]] = None
    acquisition_function: AnyMultiObjectiveAcquisitionFunction = Field(
        default_factory=lambda: qLogNEHVI()
    )

    @model_validator(mode="after")
    def validate_ref_point(self):
        """Validate that the provided refpoint matches the provided domain."""
        if self.ref_point is None:
            return self
        keys = self.domain.outputs.get_keys_by_objective(
            [MaximizeObjective, MinimizeObjective, CloseToTargetObjective]
        )
        if sorted(keys) != sorted(self.ref_point.keys()):
            raise ValueError(
                f"Provided refpoint do not match the domain, expected keys: {keys}"
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
        return my_type in [
            MaximizeObjective,
            MinimizeObjective,
            MinimizeSigmoidObjective,
            MaximizeSigmoidObjective,
            TargetObjective,
            CloseToTargetObjective,
        ]
