from typing import Literal, Type

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
