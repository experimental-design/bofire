from typing import Literal, Type, Union

from pydantic import Field

from bofire.data_models.acquisition_functions.api import qEI, qLogEI, qLogNEI, qNEI
from bofire.data_models.constraints.api import Constraint, InterpointConstraint
from bofire.data_models.features.api import Feature
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


class QparegoStrategy(MultiobjectiveStrategy):
    type: Literal["QparegoStrategy"] = "QparegoStrategy"
    acquisition_function: Union[qEI, qLogEI, qLogNEI, qNEI] = Field(
        default_factory=lambda: qNEI(),
    )

    @classmethod
    def is_feature_implemented(cls, my_type: Type[Feature]) -> bool:
        return True

    @classmethod
    def is_objective_implemented(cls, my_type: Type[Objective]) -> bool:
        if my_type not in [
            MaximizeObjective,
            MinimizeObjective,
            TargetObjective,
            MinimizeSigmoidObjective,
            MaximizeSigmoidObjective,
            CloseToTargetObjective,
        ]:
            return False
        return True

    def is_constraint_implemented(self, my_type: Type[Constraint]) -> bool:
        """Method to check if a specific constraint type is implemented for the strategy

        Args:
            my_type (Type[Constraint]): Constraint class

        Returns:
            bool: True if the constraint type is valid for the strategy chosen, False otherwise

        """
        if my_type == InterpointConstraint:
            return False

        return super().is_constraint_implemented(my_type)
