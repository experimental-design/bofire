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
