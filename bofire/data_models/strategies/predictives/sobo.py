from typing import Literal, Type

from pydantic import validator

from bofire.data_models.acquisition_functions.api import AnyAcquisitionFunction
from bofire.data_models.constraints.api import (
    Constraint,
    NonlinearEqualityConstraint,
    NonlinearInequalityConstraint,
)
from bofire.data_models.features.api import Feature
from bofire.data_models.objectives.api import BotorchConstrainedObjective, Objective
from bofire.data_models.strategies.predictives.botorch import BotorchStrategy


class SoboBaseStrategy(BotorchStrategy):
    acquisition_function: AnyAcquisitionFunction

    @classmethod
    def is_constraint_implemented(cls, my_type: Type[Constraint]) -> bool:
        """Method to check if a specific constraint type is implemented for the strategy

        Args:
            my_type (Type[Constraint]): Constraint class

        Returns:
            bool: True if the constraint type is valid for the strategy chosen, False otherwise
        """
        if my_type in [NonlinearInequalityConstraint, NonlinearEqualityConstraint]:
            return False
        return True

    @classmethod
    def is_feature_implemented(cls, my_type: Type[Feature]) -> bool:
        """Method to check if a specific feature type is implemented for the strategy

        Args:
            my_type (Type[Feature]): Feature class

        Returns:
            bool: True if the feature type is valid for the strategy chosen, False otherwise
        """
        return True

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
            len(v.outputs.get_by_objective(excludes=BotorchConstrainedObjective))
            - len(v.outputs.get_by_objective(includes=None, excludes=Objective))
        ) > 1:
            raise ValueError(
                "SOBO strategy can only deal with one no-constraint objective."
            )
        return v


class AdditiveSoboStrategy(SoboBaseStrategy):
    type: Literal["AdditiveSoboStrategy"] = "AdditiveSoboStrategy"
    use_output_constraints: bool = True


class MultiplicativeSoboStrategy(SoboBaseStrategy):
    type: Literal["MultiplicativeSoboStrategy"] = "MultiplicativeSoboStrategy"
