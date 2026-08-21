from typing import Literal

from pydantic import PositiveFloat, PositiveInt

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

    Attributes:
        lambda_cost: Exchange rate between cost and improvement. Higher values
            favour earlier stopping (require higher improvement-to-cost ratio
            to continue). Default ``1.0``.
        cost_column: Name of the column in the experiments DataFrame that
            records the cost of each experiment. When set, the mean of past
            costs is used as the cost estimate. Takes priority over
            ``cost_value``.
        cost_value: Fixed cost per experiment used when ``cost_column`` is not
            provided. Default ``1.0``.
        alpha: Exponent applied to the cost in the LogEIPC formula. ``1.0``
            (default) matches the paper's primary formulation.
        min_experiments: Minimum experiments before convergence is checked.
            Default ``5``.
        n_samples: Random domain samples used to approximate the max LogEIPC.
            Default ``2000``.
        search_method: How to find the max LogEIPC — ``"sample"`` uses random
            grid search (default); ``"optimize"`` uses gradient-based search.
        cost_model: How cost is estimated — ``"mean"`` uses the running mean
            of past costs (default); ``"gp"`` fits a GP to predict cost.

    Reference:
        Xie et al. (2025): "Cost-Aware Stopping for Bayesian Optimization"
        (arXiv:2507.12453).
    """

    type: Literal["LogEipcCriterion"] = "LogEipcCriterion"
    lambda_cost: PositiveFloat = 1.0
    cost_column: str | None = None
    cost_value: PositiveFloat = 1.0
    alpha: PositiveFloat = 1.0
    min_experiments: PositiveInt = 5
    n_samples: PositiveInt = 2000
    search_method: Literal["sample", "optimize"] = "sample"
    cost_model: Literal["mean", "gp"] = "mean"

    @classmethod
    def is_applicable_to_multiobjective(cls) -> bool:
        return False

    @classmethod
    def is_applicable_to_singleobjective(cls) -> bool:
        return True

    @classmethod
    def is_applicable_to_objective_free(cls) -> bool:
        return False
