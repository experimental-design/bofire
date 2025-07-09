from typing import Annotated, Dict, List, Literal, Optional, Type, Union

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
    delta: float = 1e-6
    transform_range: Optional[Bounds] = None


class SpaceFillingCriterion(OptimalityCriterion):
    type: Literal["SpaceFillingCriterion"] = "SpaceFillingCriterion"  # type: ignore
    sampling_fraction: Annotated[float, Field(gt=0, lt=1)] = 0.3


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


class IOptimalityCriterion(DoEOptimalityCriterion):
    type: Literal["IOptimalityCriterion"] = "IOptimalityCriterion"  # type: ignore
    n_space_filling_points: Optional[int] = None
    ipopt_options: Optional[Dict] = None


AnyDoEOptimalityCriterion = Union[
    KOptimalityCriterion,
    GOptimalityCriterion,
    AOptimalityCriterion,
    EOptimalityCriterion,
    DOptimalityCriterion,
    IOptimalityCriterion,
]

AnyOptimalityCriterion = Union[
    AnyDoEOptimalityCriterion,
    SpaceFillingCriterion,
]


class DoEStrategy(Strategy):
    type: Literal["DoEStrategy"] = "DoEStrategy"  # type: ignore

    criterion: AnyOptimalityCriterion = Field(
        default_factory=lambda: DOptimalityCriterion(formula="fully-quadratic")
    )

    verbose: bool = False  # get rid of this at a later stage
    ipopt_options: Optional[Dict] = None
    scip_params: Optional[Dict] = None
    use_hessian: bool = False
    use_cyipopt: Optional[bool] = None
    sampling: Optional[List[List[float]]] = None
    return_fixed_candidates: bool = False

    """ Datamodel for strategy for design of experiments. This strategy is used to generate a set of
    experiments for a given domain.

    Args:
        criterion: object indicating which criterion function to use.
        verbose: Should optimization be verbose?
        ipopt_options: options for IPOPT. For more information see [this link](https://coin-or.github.io/Ipopt/OPTIONS.html)
        scip_params: Parameters for SCIP solver used when generating DoEs with discrete variables.
        use_hessian: If True, the hessian of the objective function is used. Default is False.
        use_cyipopt: If True, cyipopt is used, otherwise scipy.minimize(). Default is None.
            If None, cyipopt is used if available.
        sampling: dataframe containing the initial guess.
        return_fixed_candidates: Should fixed candidates be also returned?
    """

    def is_constraint_implemented(self, my_type: Type[Constraint]) -> bool:
        return True

    @classmethod
    def is_feature_implemented(cls, my_type: Type[Feature]) -> bool:
        if my_type in [MolecularInput]:
            return False
        return True

    @classmethod
    def is_objective_implemented(cls, my_type: Type[Objective]) -> bool:
        return True
