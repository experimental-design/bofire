from typing import Literal, Type

from bofire.data_models.constraints.api import Constraint, NChooseKConstraint
from bofire.data_models.features.api import Feature
from bofire.data_models.objectives.api import (
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
    """
    This class defines a strategy for multi-objective optimization using the QParego algorithm.

    Attributes:
    type (Literal["QparegoStrategy"]): A literal indicating the type of strategy, which is "QparegoStrategy".

    Methods:
    is_constraint_implemented(my_type: Type[Constraint]) -> bool:
    Checks if the given constraint type is implemented in this strategy.
    """

    type: Literal["QparegoStrategy"] = "QparegoStrategy"

    @classmethod
    def is_constraint_implemented(cls, my_type: Type[Constraint]) -> bool:
        if my_type == NChooseKConstraint:
            return False
        return True

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
        ]:
            return False
        return True
