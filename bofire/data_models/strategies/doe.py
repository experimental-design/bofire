from typing import Literal, Union, Type
from bofire.data_models.strategies.strategy import Strategy
from bofire.data_models.constraints.api import (
    Constraint,
    NChooseKConstraint,
    NonlinearEqualityConstraint,
)
from bofire.data_models.features.api import Feature
from bofire.data_models.objectives.api import Objective


class DoEStrategy(Strategy):
    type: Literal["DoEStrategy"] = "DoEStrategy"
    formula: Union[
        Literal[
            "linear",
            "linear-and-quadratic",
            "linear-and-interactions",
            "fully-quadratic",
        ],
        str,
    ]

    @classmethod
    def is_constraint_implemented(cls, my_type: Type[Constraint]) -> bool:
        if my_type in [NChooseKConstraint, NonlinearEqualityConstraint]:
            return False
        return True

    @classmethod
    def is_feature_implemented(cls, my_type: Type[Feature]) -> bool:
        return True

    @classmethod
    def is_objective_implemented(cls, my_type: Type[Objective]) -> bool:
        return True
