r"""Functional convergence evaluation for the objective improvement criterion.

The evaluator is a pure function of the criterion and the strategy's *recorded
history*: it must not keep internal state between ``has_converged`` calls. The
signal is derived from ``strategy.experiments`` (which accumulate and are never
reset), so a strategy reconstructed by replaying ``tell`` reaches the same
result.

Math:
    Each recorded experiment is mapped to a scalar reward by evaluating the
    domain objectives (higher is better) and summing the per-output
    desirabilities. Let :math:`r_1, \dots, r_N` be these rewards in chronological
    order and

    .. math:: b_k = \max_{i \le k} r_i

    the best reward within the first :math:`k` experiments. Convergence holds
    once the best reward improved by less than ``min_improvement`` over the last
    ``n_lookback`` experiments,

    .. math:: b_N - b_{N - \text{n\_lookback}} < \text{min\_improvement}.
"""

from typing import TYPE_CHECKING

import numpy as np

from bofire.data_models.strategies.convergence_criteria.api import (
    ObjectiveImprovementCriterion,
)


if TYPE_CHECKING:
    from bofire.strategies.strategy import Strategy
    from bofire.surrogates.botorch_surrogates import BotorchSurrogates


def evaluate_objective_improvement_criterion(
    criterion: ObjectiveImprovementCriterion,
    strategy: "Strategy",
    surrogates: "BotorchSurrogates | None",
) -> bool:
    """Evaluate whether the best objective stopped improving.

    Args:
        criterion: The convergence criterion data model with its parameters.
        strategy: The functional strategy providing the recorded experiments.
        surrogates: The fitted surrogate model(s) of the strategy. Not used by
            this criterion, as it is evaluated on the observed data alone.

    Returns:
        bool: True if the best reward improved by less than ``min_improvement``
        over the last ``n_lookback`` experiments, False otherwise (including when
        there are not yet more than ``n_lookback`` experiments).
    """
    experiments = strategy.experiments
    if experiments is None:
        return False

    # Map each experiment to a scalar reward via the domain objectives
    # (higher is better) and sum the per-output desirabilities.
    desirabilities = strategy.domain.outputs(experiments)
    rewards = desirabilities.sum(axis=1).to_numpy(dtype=float)
    rewards = rewards[~np.isnan(rewards)]

    n = rewards.shape[0]
    # Need at least one experiment before the lookback window to form a baseline.
    if n <= criterion.n_lookback:
        return False

    best_now = float(np.max(rewards))
    best_before = float(np.max(rewards[: n - criterion.n_lookback]))
    improvement = best_now - best_before
    return bool(improvement < criterion.min_improvement)
