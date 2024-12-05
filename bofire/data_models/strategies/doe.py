from typing import Literal, Optional, Type, Union

from formulaic import Formula
from formulaic.errors import FormulaSyntaxError
from pydantic import Field, field_validator

from bofire.data_models.base import BaseModel
from bofire.data_models.constraints.api import Constraint
from bofire.data_models.features.api import Feature, MolecularInput
from bofire.data_models.objectives.api import Objective
from bofire.data_models.strategies.strategy import Strategy
from bofire.data_models.types import Bounds


PREDEFINED_MODEL_TYPES = Literal[
    "linear",
    "linear-and-quadratic",
    "linear-and-interactions",
    "fully-quadratic",
]


class OptimalityCriterion(BaseModel):
    type: str
    transform_range: Optional[Bounds] = None


class SpaceFillingCriterion(OptimalityCriterion):
    type: Literal["SpaceFillingCriterion"] = "SpaceFillingCriterion"  # type: ignore


class DoEOptimalityCriterion(OptimalityCriterion):
    type: str
    formula: Union[
        PREDEFINED_MODEL_TYPES,
        str,
    ]
    """
    model_type (str, Formula): keyword or formulaic Formula describing the model. Known keywords
    are "linear", "linear-and-interactions", "linear-and-quadratic", "fully-quadratic".
    """

    @field_validator("formula")
    @classmethod
    def validate_formula(cls, formula: str) -> str:
        if formula not in PREDEFINED_MODEL_TYPES.__args__:  # type: ignore
            # check that it is a valid formula
            try:
                Formula(formula)
            except FormulaSyntaxError:
                raise ValueError(f"Invalid formula: {formula}")
        return formula


class DOptimalityCriterion(DoEOptimalityCriterion):
    type: Literal["DOptimalityCriterion"] = "DOptimalityCriterion"  # type: ignore


class EOptimalityCriterion(DoEOptimalityCriterion):
    type: Literal["EOptimalityCriterion"] = "EOptimalityCriterion"  # type: ignore


class AOptimalityCriterion(DoEOptimalityCriterion):
    type: Literal["AOptimalityCriterion"] = "AOptimalityCriterion"  # type: ignore


class GOptimalityCriterion(DoEOptimalityCriterion):
    type: Literal["GOptimalityCriterion"] = "GOptimalityCriterion"  # type: ignore


class KOptimalityCriterion(DoEOptimalityCriterion):
    type: Literal["KOptimalityCriterion"] = "KOptimalityCriterion"  # type: ignore


AnyDoEOptimalityCriterion = Union[
    KOptimalityCriterion,
    GOptimalityCriterion,
    AOptimalityCriterion,
    EOptimalityCriterion,
    DOptimalityCriterion,
]

AnyOptimalityCriterion = Union[
    AnyDoEOptimalityCriterion,
    SpaceFillingCriterion,
]


class DoEStrategy(Strategy):
    type: Literal["DoEStrategy"] = "DoEStrategy"  # type: ignore

    criterion: AnyDoEOptimalityCriterion = Field(
        default_factory=lambda: DOptimalityCriterion(formula="fully-quadratic")
    )
    optimization_strategy: Literal[
        "default",
        "exhaustive",
        "branch-and-bound",
        "partially-random",
        "relaxed",
        "iterative",
    ] = "default"

    verbose: bool = False  # get rid of this at a later stage

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
