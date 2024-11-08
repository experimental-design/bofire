from typing import Literal, Optional, Type, Union

from bofire.data_models.constraints.api import Constraint
from bofire.data_models.features.api import Feature, MolecularInput
from bofire.data_models.objectives.api import Objective
from bofire.data_models.strategies.strategy import Strategy
from bofire.data_models.types import Bounds
from bofire.strategies.enum import OptimalityCriterionEnum


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
    optimization_strategy: Literal[
        "default",
        "exhaustive",
        "branch-and-bound",
        "partially-random",
        "relaxed",
        "iterative",
    ] = "default"

    verbose: bool = False

    objective: OptimalityCriterionEnum = OptimalityCriterionEnum.D_OPTIMALITY

    transform_range: Optional[Bounds] = None

    @classmethod
    def is_constraint_implemented(cls, my_type: Type[Constraint]) -> bool:
        return True

    @classmethod
    def is_feature_implemented(cls, my_type: Type[Feature]) -> bool:
        if my_type in [MolecularInput]:
            return False
        return True

    @classmethod
    def is_objective_implemented(cls, my_type: Type[Objective]) -> bool:
        return True
