from typing import Literal, Tuple, Type, Union

from pydantic import field_validator

from bofire.data_models.constraints.api import Constraint
from bofire.data_models.features.api import (
    Feature,
    MolecularInput,
)
from bofire.data_models.objectives.api import Objective
from bofire.data_models.strategies.strategy import Strategy
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

    transform_range: Union[None, Tuple[float, float]] = None

    @field_validator("transform_range")
    @classmethod
    def validate_feature_range(cls, value):
        if value is not None:
            if not isinstance(value, tuple) or len(value) != 2:
                raise ValueError("feature_range must be a tuple of length 2")
            if value[0] >= value[1]:
                raise ValueError(
                    "feature_range[0] must be smaller than feature_range[1]"
                )
            return value

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
