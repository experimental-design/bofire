from typing import Literal, Optional, Type

from pydantic import Field, validator

from bofire.data_models.acquisition_functions.api import (
    AnySingleObjectiveAcquisitionFunction,
    qLogNEI,
)
from bofire.data_models.features.api import CategoricalOutput, Feature
from bofire.data_models.objectives.api import ConstrainedObjective, Objective
from bofire.data_models.strategies.predictives.botorch import BotorchStrategy


class SoboBaseStrategy(BotorchStrategy):
    acquisition_function: AnySingleObjectiveAcquisitionFunction = Field(
        default_factory=lambda: qLogNEI()
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


class SoboStrategy(SoboBaseStrategy):
    type: Literal["SoboStrategy"] = "SoboStrategy"

    @validator("domain")
    def validate_is_singleobjective(cls, v, values):
        if len(v.outputs) == 1:
            return v
        if (
            len(v.outputs.get_by_objective(excludes=ConstrainedObjective))
            - len(v.outputs.get_by_objective(includes=None, excludes=Objective))
        ) > 1:
            raise ValueError(
                "SOBO strategy can only deal with one no-constraint objective."
            )
        return v


class AdditiveSoboStrategy(SoboBaseStrategy):
    type: Literal["AdditiveSoboStrategy"] = "AdditiveSoboStrategy"
    use_output_constraints: bool = True

    @validator("domain")
    def validate_is_multiobjective(cls, v, values):
        if (len(v.outputs.get_by_objective(Objective))) < 2:
            raise ValueError(
                "Additive SOBO strategy requires at least 2 outputs with objectives. Consider SOBO strategy instead."
            )
        return v


class MultiplicativeSoboStrategy(SoboBaseStrategy):
    type: Literal["MultiplicativeSoboStrategy"] = "MultiplicativeSoboStrategy"

    @validator("domain")
    def validate_is_multiobjective(cls, v, values):
        if (len(v.outputs.get_by_objective(Objective))) < 2:
            raise ValueError(
                "Multiplicative SOBO strategy requires at least 2 outputs with objectives. Consider SOBO strategy instead."
            )
        return v


class CustomSoboStrategy(SoboBaseStrategy):
    type: Literal["CustomSoboStrategy"] = "CustomSoboStrategy"
    use_output_constraints: bool = True
    dump: Optional[str] = None
