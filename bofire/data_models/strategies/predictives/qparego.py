from typing import Literal, Type, Union

from pydantic import Field

from bofire.data_models.acquisition_functions.api import qEI, qLogEI, qLogNEI, qNEI
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


class QparegoStrategy(MultiobjectiveStrategy):
    type: Literal["QparegoStrategy"] = "QparegoStrategy"
    acquisition_function: Union[qEI, qLogEI, qLogNEI, qNEI] = Field(
        default_factory=lambda: qNEI()
    )

    @classmethod
    def is_feature_implemented(cls, my_type: Type[Feature]) -> bool:
        if my_type not in [CategoricalOutput]:
            return True
        return False

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
