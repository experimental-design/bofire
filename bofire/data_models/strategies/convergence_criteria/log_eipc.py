from typing import Literal

from pydantic import Field, PositiveFloat, PositiveInt

from bofire.data_models.strategies.convergence_criteria.convergence_criterion import (
    ConvergenceCriterion,
)


class LogEipcCriterion(ConvergenceCriterion):
    """Cost-aware convergence criterion (Xie et al., 2025).

    The optimization is considered converged when the maximum log expected
    improvement-per-cost over the domain drops to zero or below — i.e. no
    unevaluated point's expected improvement is worth its evaluation cost:

        converged when  max_x [ LogEI(x) - alpha * log(c(x)) - log(lambda_cost) ] <= 0

    Ideal for chemical experiments where reagent, time, or equipment costs
    matter.

    Requires a fitted GP-based strategy (e.g. ``SoboStrategy``).
    Single-objective only.

    Reference:
        Xie et al. (2025): "Cost-Aware Stopping for Bayesian Optimization"
        (arXiv:2507.12453).
    """

    type: Literal["LogEipcCriterion"] = "LogEipcCriterion"
    lambda_cost: PositiveFloat = Field(
        default=1.0,
        description="Exchange rate between cost and improvement; higher values "
        "favour earlier stopping.",
    )
    cost_column: str | None = Field(
        default=None,
        description="Experiments column recording the cost of each experiment; "
        "when set, past costs provide the cost estimate (see `cost_model`).",
    )
    cost_value: PositiveFloat = Field(
        default=1.0,
        description="Fixed cost per experiment used when `cost_column` is not "
        "provided.",
    )
    alpha: PositiveFloat = Field(
        default=1.0,
        description="Exponent applied to the cost in the LogEIPC formula.",
    )
    min_experiments: PositiveInt = Field(
        default=5,
        description="Minimum experiments before convergence is checked.",
    )
    n_samples: PositiveInt = Field(
        default=2000,
        description="Random domain samples used to approximate the maximum " "LogEIPC.",
    )
    search_method: Literal["sample", "optimize"] = Field(
        default="sample",
        description="How the maximum LogEIPC is found: random sampling or "
        "gradient-based search.",
    )
    cost_model: Literal["mean", "gp"] = Field(
        default="mean",
        description="Cost estimate from the running mean of past costs, or "
        "from a GP fit to them.",
    )

    @classmethod
    def is_applicable_to_multiobjective(cls) -> bool:
        return False

    @classmethod
    def is_applicable_to_singleobjective(cls) -> bool:
        return True

    @classmethod
    def is_applicable_to_objective_free(cls) -> bool:
        return False
