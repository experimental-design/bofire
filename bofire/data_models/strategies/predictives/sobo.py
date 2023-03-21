from typing import Literal, Type

from bofire.data_models.acquisition_functions.api import AnyAcquisitionFunction
from bofire.data_models.constraints.api import Constraint
from bofire.data_models.features.api import Feature
from bofire.data_models.objectives.api import Objective
from bofire.data_models.strategies.predictives.botorch import BotorchStrategy


class SoboStrategy(BotorchStrategy):
    """
    A strategy class that implements the Sobol sequence method for Bayesian optimization using Botorch.

    Attributes:
        type (Literal["SoboStrategy"]): The string literal representing the type of the strategy.
        acquisition_function (AnyAcquisitionFunction): The acquisition function used for optimizing the objective.

    Methods:
        is_constraint_implemented(cls, my_type: Type[Constraint]) -> bool:
            Check if a specific constraint type is implemented for the strategy.

        is_feature_implemented(cls, my_type: Type[Feature]) -> bool:
            Check if a specific feature type is implemented for the strategy.

        is_objective_implemented(cls, my_type: Type[Objective]) -> bool:
            Check if an objective type is implemented for the strategy.
    """

    type: Literal["SoboStrategy"] = "SoboStrategy"
    acquisition_function: AnyAcquisitionFunction

    @classmethod
    def is_constraint_implemented(cls, my_type: Type[Constraint]) -> bool:
        """Method to check if a specific constraint type is implemented for the strategy

        Args:
            my_type (Type[Constraint]): Constraint class

        Returns:
            bool: True if the constraint type is valid for the strategy chosen, False otherwise
        """
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


class AdditiveSoboStrategy(SoboStrategy):
    """
    A subclass of SoboStrategy that implements an additive version of the Sobol sequence-based strategy for Bayesian optimization.

    Attributes:
    type (Literal["AdditiveSoboStrategy"]): A string literal indicating the type of the strategy, which is "AdditiveSoboStrategy".
    use_output_constraints (bool): A boolean flag indicating whether or not to enforce output constraints during optimization.
    """

    type: Literal["AdditiveSoboStrategy"] = "AdditiveSoboStrategy"
    use_output_constraints: bool = True


class MultiplicativeSoboStrategy(SoboStrategy):
    """
    A subclass of SoboStrategy that implements an multiplicative version of the Sobol sequence-based strategy for Bayesian optimization.

    Attributes:
    type (Literal["AdditiveSoboStrategy"]): A string literal indicating the type of the strategy, which is "MultiplicativeSoboStrategy".
    """

    type: Literal["MultiplicativeSoboStrategy"] = "MultiplicativeSoboStrategy"
