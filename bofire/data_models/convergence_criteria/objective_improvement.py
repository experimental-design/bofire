from typing import Literal

from pydantic import PositiveFloat, PositiveInt

from bofire.data_models.convergence_criteria.convergence_criterion import (
    ConvergenceCriterion,
)


class ObjectiveImprovementCriterion(ConvergenceCriterion):
    r"""Convergence based on the improvement of the best observed objective.

    Each recorded experiment is mapped to a scalar reward by evaluating the
    domain objectives (higher is better) and summing the per-output
    desirabilities. Let :math:`r_1, \dots, r_N` be these rewards in chronological
    order and

    .. math:: b_k = \max_{i \le k} r_i

    the best reward observed within the first :math:`k` experiments. The
    optimization is considered converged once the best reward has improved by
    less than ``min_improvement`` over the last ``n_lookback`` experiments,

    .. math:: b_N - b_{N - \text{n\_lookback}} < \text{min\_improvement}.

    At least ``n_lookback + 1`` experiments are required; otherwise the strategy
    is not considered converged.

    Attributes:
        min_improvement: Minimal improvement of the best observed objective that
            is still considered relevant.
        n_lookback: Number of most recent experiments over which the improvement
            is evaluated.
    """

    type: Literal["ObjectiveImprovementCriterion"] = "ObjectiveImprovementCriterion"
    min_improvement: PositiveFloat
    n_lookback: PositiveInt
