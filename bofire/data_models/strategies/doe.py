from typing import Literal, Type, Union

from bofire.data_models.constraints.api import Constraint
from bofire.data_models.features.api import (
    Feature,
    MolecularInput,
)
from bofire.data_models.objectives.api import Objective
from bofire.data_models.strategies.strategy import Strategy


class DoEStrategy(Strategy):
    """

    Attributes:
        optimization_strategy (Literal["default", "exhaustive", "branch-and-bound", "random-nchoosek-realization", "relaxed"]):
            if not specified uses "default".
            "default": if no CategoricalInputs, DiscreteInputs or NChooseKConstraints are provided in the domain.
                It is solved as a continuous problem. If any of those are provided branch-and-bound is used.
            "exhaustive": tries every possible combination of CategoricalInputs, DiscreteInputs and NChooseKConstraints.
                If none of those are provided, it is the same as "default"
            "branch-and-bound": Uses the Branch-and-Bound-Algorithm to find a solution. Same as "default" if no
                CategoricalInputs, DiscreteInputs or NChooseKConstraints are provided in the domain.
            "random-nchoosek-realization": Solves the NChooseKConstraints by finding a valid (none optimal) solution.
                If CategoricalInputs and DiscreteInputs are provided those are solved with Branch-and-Bound. If neither of
                CategoricalInputs, DiscreteInputs or NChooseKConstraints are provided, it is the same as "default"
            "relaxed": tries to solve the problem on a relaxed domain. This may not result in a valid solution and might
                thereby fail with an error.
    """

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
        "random-nchoosek-realization",
        "relaxed",
    ] = "default"

    verbose: bool = False

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
