import warnings
from typing import Annotated, Any, Dict, List, Literal, Optional, Type, Union

from formulaic import Formula
from formulaic.errors import FormulaSyntaxError
from pydantic import Field, field_validator, model_validator

from bofire.data_models.base import BaseModel
from bofire.data_models.constraints.api import Constraint
from bofire.data_models.features.api import Feature
from bofire.data_models.objectives.api import Objective
from bofire.data_models.strategies.strategy import Strategy
from bofire.data_models.types import Bounds
from bofire.data_models.unions import tagged_union


def _pop_legacy_ipopt_options(data: dict) -> dict:
    """Move a legacy ``ipopt_options`` entry to ``optimizer_options``."""
    if "ipopt_options" in data:
        legacy = data.pop("ipopt_options")
        warnings.warn(
            "`ipopt_options` is deprecated; use `optimizer_options`.",
            DeprecationWarning,
            stacklevel=4,
        )
        data.setdefault("optimizer_options", legacy)
    return data


PREDEFINED_MODEL_TYPES = Literal[
    "linear",
    "linear-and-quadratic",
    "linear-and-interactions",
    "fully-quadratic",
]


class OptimalityCriterion(BaseModel):
    type: Any
    delta: float = 1e-6
    transform_range: Optional[Bounds] = None


class SpaceFillingCriterion(OptimalityCriterion):
    type: Literal["SpaceFillingCriterion"] = "SpaceFillingCriterion"
    sampling_fraction: Annotated[float, Field(gt=0, lt=1)] = 0.3


class DoEOptimalityCriterion(OptimalityCriterion):
    type: Any
    formula: Union[PREDEFINED_MODEL_TYPES, str]
    """
    model_type (str, Formula): keyword or formulaic Formula describing the model. Known keywords
    are "linear", "linear-and-interactions", "linear-and-quadratic", "fully-quadratic".
    """

    @field_validator("formula")
    @classmethod
    def validate_formula(cls, formula: str) -> str:
        if formula not in PREDEFINED_MODEL_TYPES.__args__:
            # check that it is a valid formula
            try:
                Formula(formula)
            except FormulaSyntaxError:
                raise ValueError(f"Invalid formula: {formula}")
        return formula


class DOptimalityCriterion(DoEOptimalityCriterion):
    type: Literal["DOptimalityCriterion"] = "DOptimalityCriterion"


class EOptimalityCriterion(DoEOptimalityCriterion):
    type: Literal["EOptimalityCriterion"] = "EOptimalityCriterion"


class AOptimalityCriterion(DoEOptimalityCriterion):
    type: Literal["AOptimalityCriterion"] = "AOptimalityCriterion"


class GOptimalityCriterion(DoEOptimalityCriterion):
    type: Literal["GOptimalityCriterion"] = "GOptimalityCriterion"


class KOptimalityCriterion(DoEOptimalityCriterion):
    type: Literal["KOptimalityCriterion"] = "KOptimalityCriterion"


class IOptimalityCriterion(DoEOptimalityCriterion):
    type: Literal["IOptimalityCriterion"] = "IOptimalityCriterion"
    n_space_filling_points: Optional[int] = None
    optimizer_options: Optional[Dict] = Field(
        default=None,
        description=(
            "Options for the solver that places the space-filling points, e.g. "
            "`max_iter` or `print_level`. The solver is the strategy's `optimizer`."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def validate_legacy_ipopt_options(cls, data):
        """Accept the deprecated ``ipopt_options`` as ``optimizer_options``."""
        if not isinstance(data, dict):
            return data
        return _pop_legacy_ipopt_options({**data})


AnyDoEOptimalityCriterion = tagged_union(
    KOptimalityCriterion,
    GOptimalityCriterion,
    AOptimalityCriterion,
    EOptimalityCriterion,
    DOptimalityCriterion,
    IOptimalityCriterion,
)

AnyOptimalityCriterion = tagged_union(
    KOptimalityCriterion,
    GOptimalityCriterion,
    AOptimalityCriterion,
    EOptimalityCriterion,
    DOptimalityCriterion,
    IOptimalityCriterion,
    SpaceFillingCriterion,
)


class DoEStrategy(Strategy):
    type: Literal["DoEStrategy"] = "DoEStrategy"

    criterion: AnyOptimalityCriterion = Field(
        default_factory=lambda: DOptimalityCriterion(formula="fully-quadratic")
    )

    verbose: bool = False  # get rid of this at a later stage
    optimizer: Literal["pounce", "ipopt", "scipy"] = Field(
        default="pounce",
        description=(
            "NLP solver for the continuous design problem: `pounce` (default), "
            "`ipopt` (requires cyipopt) or `scipy` (SLSQP, slow beyond a few "
            "hundred variables). Raises if the chosen solver is not installed."
        ),
    )
    optimizer_options: Optional[Dict] = Field(
        default=None,
        description=(
            "Options passed to the solver, overriding BoFire's defaults. For "
            "`pounce` and `ipopt` these are Ipopt options such as `max_iter`, "
            "`tol` or `print_level`; for `scipy` only `max_iter` and "
            "`print_level` are used."
        ),
    )
    scip_params: Optional[Dict] = None
    sampling: Optional[List[List[float]]] = None
    return_fixed_candidates: bool = False

    """ Datamodel for strategy for design of experiments. This strategy is used to generate a set of
    experiments for a given domain.

    Args:
        criterion: object indicating which criterion function to use.
        verbose: Should optimization be verbose?
        scip_params: Parameters for SCIP solver used when generating DoEs with discrete variables.
        sampling: dataframe containing the initial guess.
        return_fixed_candidates: Should fixed candidates be also returned?
    """

    @model_validator(mode="before")
    @classmethod
    def validate_legacy_solver_fields(cls, data):
        """Accept the deprecated ``use_cyipopt``, ``ipopt_options`` and
        ``use_hessian`` fields.

        ``use_cyipopt=True`` maps to ``optimizer="ipopt"``, ``False`` to
        ``"scipy"`` and ``None`` to the default; ``ipopt_options`` becomes
        ``optimizer_options``. Explicit new fields win. ``use_hessian`` is
        ignored. Each legacy field emits a ``DeprecationWarning``.
        """
        if not isinstance(data, dict):
            return data
        data = _pop_legacy_ipopt_options({**data})
        if "use_cyipopt" in data:
            use_cyipopt = data.pop("use_cyipopt")
            warnings.warn(
                "`use_cyipopt` is deprecated; use `optimizer` "
                "('pounce', 'ipopt' or 'scipy').",
                DeprecationWarning,
                stacklevel=3,
            )
            if use_cyipopt is not None:
                data.setdefault("optimizer", "ipopt" if use_cyipopt else "scipy")
        if "use_hessian" in data:
            data.pop("use_hessian")
            warnings.warn(
                "`use_hessian` is deprecated and ignored.",
                DeprecationWarning,
                stacklevel=3,
            )
        return data

    def is_constraint_implemented(self, my_type: Type[Constraint]) -> bool:
        return True

    @classmethod
    def is_feature_implemented(cls, my_type: Type[Feature]) -> bool:
        return True

    @classmethod
    def is_objective_implemented(cls, my_type: Type[Objective]) -> bool:
        return True
