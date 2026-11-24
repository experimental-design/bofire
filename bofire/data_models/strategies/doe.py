from typing import Any, Dict, Literal, Optional, Type, Union

from formulaic import Formula
from formulaic.errors import FormulaSyntaxError
from pydantic import Field, field_validator

from bofire.data_models.base import BaseModel
from bofire.data_models.constraints.api import Constraint
from bofire.data_models.features.api import Feature
from bofire.data_models.objectives.api import Objective
from bofire.data_models.strategies.strategy import Strategy
from bofire.data_models.types import Bounds
from bofire.data_models.unions import tagged_union


DoEOptimizer = Literal["pounce", "ipopt", "scipy"]
DEFAULT_DOE_OPTIMIZER: DoEOptimizer = "pounce"

PREDEFINED_MODEL_TYPES = Literal[
    "linear",
    "linear-and-quadratic",
    "linear-and-interactions",
    "fully-quadratic",
]


class OptimalityCriterion(BaseModel):
    """Objective that a design of experiments optimizes."""

    type: Any
    delta: float = Field(
        default=1e-6,
        description=(
            "Regularization added to the diagonal of the information matrix "
            "`X^T X`, so it stays invertible for designs that do not yet support "
            "every model term. Keep it small relative to the matrix entries."
        ),
    )
    transform_range: Optional[Bounds] = Field(
        default=None,
        description=(
            "Range, e.g. `[-1, 1]`, onto which every input is mapped linearly from "
            "its bounds before the criterion is evaluated (coded units). `None` "
            "evaluates in the original units. Not supported by I-optimality."
        ),
    )


class SpaceFillingCriterion(OptimalityCriterion):
    """Spreads the experiments over the design space without assuming a model:
    maximizes the sum of the smallest pairwise distances between the
    experiments (as many distances as there are experiments).

    Examples:
        >>> SpaceFillingCriterion()
    """

    type: Literal["SpaceFillingCriterion"] = "SpaceFillingCriterion"


class DoEOptimalityCriterion(OptimalityCriterion):
    """Model-based criterion: a function of the information matrix `X^T X` of the
    model given by `formula`, where `X` is the model matrix of the design."""

    type: Any
    formula: Union[PREDEFINED_MODEL_TYPES, str] = Field(
        description=(
            "Model whose terms the design should estimate: one of `linear`, "
            "`linear-and-interactions`, `linear-and-quadratic`, `fully-quadratic`, "
            "or a formula in Wilkinson notation over the input keys, e.g. "
            "`x1 + x2 + x1:x2`. For mixtures, where the inputs sum to a constant, "
            "drop the intercept (`-1 + ...`), otherwise the information matrix is "
            "singular."
        )
    )

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
    """D-optimal design: maximizes the determinant of `X^T X`, which minimizes the
    joint confidence region of the model coefficients. The usual default.

    Examples:
        >>> DOptimalityCriterion(formula="linear-and-interactions")
    """

    type: Literal["DOptimalityCriterion"] = "DOptimalityCriterion"


class EOptimalityCriterion(DoEOptimalityCriterion):
    """E-optimal design: maximizes the smallest eigenvalue of `X^T X`, which
    minimizes the variance of the worst-estimated combination of coefficients.

    Examples:
        >>> EOptimalityCriterion(formula="linear")
    """

    type: Literal["EOptimalityCriterion"] = "EOptimalityCriterion"


class AOptimalityCriterion(DoEOptimalityCriterion):
    """A-optimal design: minimizes the trace of `(X^T X)^-1`, the average variance
    of the model coefficients.

    Examples:
        >>> AOptimalityCriterion(formula="linear")
    """

    type: Literal["AOptimalityCriterion"] = "AOptimalityCriterion"


class GOptimalityCriterion(DoEOptimalityCriterion):
    """G-optimal design: minimizes the largest prediction variance at the design
    points (the maximum diagonal entry of the hat matrix). Non-smooth, so the
    solver often stops at the iteration limit.

    Examples:
        >>> GOptimalityCriterion(formula="linear")
    """

    type: Literal["GOptimalityCriterion"] = "GOptimalityCriterion"


class KOptimalityCriterion(DoEOptimalityCriterion):
    """K-optimal design: minimizes the condition number of `X^T X`. The objective
    spans many orders of magnitude, so results vary strongly with the start.

    Examples:
        >>> KOptimalityCriterion(formula="linear")
    """

    type: Literal["KOptimalityCriterion"] = "KOptimalityCriterion"


class IOptimalityCriterion(DoEOptimalityCriterion):
    """I-optimal design: minimizes the average prediction variance over the design
    space, which suits designs meant for prediction rather than coefficient
    estimation. The design space is represented by an equidistant grid or, with
    equality constraints, by space-filling points.

    Examples:
        >>> IOptimalityCriterion(formula="fully-quadratic")
    """

    type: Literal["IOptimalityCriterion"] = "IOptimalityCriterion"
    n_space_filling_points: Optional[int] = Field(
        default=None,
        description=(
            "Number of points representing the design space when the domain has "
            "equality constraints; `None` uses ten times the number of "
            "experiments. Without equality constraints an equidistant grid is "
            "used and this has no effect."
        ),
    )
    optimizer_options: Optional[Dict] = Field(
        default=None,
        description=(
            "Options for the solver that places the space-filling points, e.g. "
            "`max_iter` or `print_level`. The solver is the strategy's `optimizer`."
        ),
    )


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
    """Generates a design of experiments that optimizes `criterion` within the
    domain's bounds and constraints. Categorical and discrete inputs are first
    relaxed to continuous ones and afterwards projected onto allowed values.
    Candidates set on the strategy are kept fixed and count towards the
    criterion. The result is a local optimum from a random start, so designs
    differ between seeds.

    Examples:
        >>> DoEStrategy(
        ...     domain=domain,
        ...     criterion=DOptimalityCriterion(formula="linear-and-interactions"),
        ... )
    """

    type: Literal["DoEStrategy"] = "DoEStrategy"

    criterion: AnyOptimalityCriterion = Field(
        default_factory=lambda: DOptimalityCriterion(formula="fully-quadratic"),
        description=(
            "Criterion the design optimizes. Defaults to D-optimality for a "
            "fully quadratic model."
        ),
    )

    optimizer: DoEOptimizer = Field(
        default=DEFAULT_DOE_OPTIMIZER,
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
    scip_params: Optional[Dict] = Field(
        default=None,
        description=(
            "Parameters for the SCIP solver that projects categorical and discrete "
            "inputs onto allowed values. `None` uses `{'numerics/feastol': 1e-8}`."
        ),
    )
    return_fixed_candidates: bool = Field(
        default=False,
        description=(
            "Whether `ask` also returns the fixed candidates set on the strategy, "
            "ahead of the new experiments. By default only the new experiments "
            "are returned."
        ),
    )

    def is_constraint_implemented(self, my_type: Type[Constraint]) -> bool:
        return True

    @classmethod
    def is_feature_implemented(cls, my_type: Type[Feature]) -> bool:
        return True

    @classmethod
    def is_objective_implemented(cls, my_type: Type[Objective]) -> bool:
        return True
