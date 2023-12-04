from typing import Literal, Type

from bofire.data_models.constraints.api import (
    Constraint,
    InterpointEqualityConstraint,
    LinearEqualityConstraint,
    LinearInequalityConstraint,
    NChooseKConstraint,
)
from bofire.data_models.features.api import Feature
from bofire.data_models.objectives.api import Objective
from bofire.data_models.strategies.strategy import Strategy


class RandomStrategy(Strategy):
    type: Literal["RandomStrategy"] = "RandomStrategy"

    @classmethod
    def is_constraint_implemented(cls, my_type: Type[Constraint]) -> bool:
        return my_type in [
            LinearInequalityConstraint,
            LinearEqualityConstraint,
            NChooseKConstraint,
            InterpointEqualityConstraint,
        ]

    @classmethod
    def is_feature_implemented(cls, my_type: Type[Feature]) -> bool:
        return True

    @classmethod
    def is_objective_implemented(cls, my_type: Type[Objective]) -> bool:
        return True
