r"""Functional convergence evaluation for the hypervolume improvement criterion.

The evaluator is a pure function of the criterion and the strategy's *recorded
history*: it must not keep internal state between ``has_converged`` calls. The
signal is derived from ``strategy.experiments`` (which accumulate and are never
reset), so a strategy reconstructed by replaying ``tell`` reaches the same
result.

Math:
    Let :math:`H_k` be the hypervolume dominated by the Pareto front of the first
    :math:`k` recorded experiments (chronological order) with respect to a common
    reference point inferred from all experiments. Since adding an experiment can
    never decrease the dominated hypervolume, :math:`H_k` is monotonically
    non-decreasing. Convergence holds once the dominated hypervolume improved by
    less than ``min_improvement`` over the last ``n_lookback`` experiments,

    .. math:: H_N - H_{N - \text{n\_lookback}} < \text{min\_improvement}.
"""

from typing import TYPE_CHECKING

import pandas as pd

from bofire.data_models.objectives.api import (
    CloseToTargetObjective,
    MaximizeObjective,
    MinimizeObjective,
)
from bofire.data_models.strategies.convergence_criteria.api import (
    HypervolumeImprovementCriterion,
)
from bofire.utils.multiobjective import (
    compute_hypervolume,
    get_pareto_front,
    infer_ref_point,
)


if TYPE_CHECKING:
    from bofire.strategies.predictives.predictive import PredictiveStrategy


def evaluate_hypervolume_improvement_criterion(
    criterion: HypervolumeImprovementCriterion,
    strategy: "PredictiveStrategy",
) -> bool:
    """Evaluate whether the dominated hypervolume stopped improving.

    Args:
        criterion: The convergence criterion data model with its parameters.
        strategy: The functional strategy providing the recorded experiments.

    Returns:
        bool: True if the dominated hypervolume improved by less than
        ``min_improvement`` over the last ``n_lookback`` experiments, False
        otherwise (including when there are not yet more than ``n_lookback``
        experiments with valid outputs, or fewer than two output objectives).
    """
    experiments = strategy.experiments
    if experiments is None:
        return False

    domain = strategy.domain
    outputs = domain.outputs.get_by_objective(
        includes=[MaximizeObjective, MinimizeObjective, CloseToTargetObjective],
    )
    # Hypervolume is only defined for at least two objectives.
    if len(outputs) < 2:
        return False

    # Restrict to experiments with valid outputs, keeping chronological order.
    valid = domain.outputs.preprocess_experiments_all_valid_outputs(
        experiments,
        output_feature_keys=outputs.get_keys(),
    )
    n = len(valid)
    # Need at least one experiment before the lookback window to form a baseline.
    if n <= criterion.n_lookback:
        return False

    # A common reference point inferred from all valid experiments makes the
    # hypervolumes of the two windows directly comparable.
    ref_point = infer_ref_point(domain, valid)

    def _hypervolume(subset: pd.DataFrame) -> float:
        pareto = get_pareto_front(domain, subset)
        return compute_hypervolume(domain, pareto, ref_point)

    hv_now = _hypervolume(valid.iloc[:n])
    hv_before = _hypervolume(valid.iloc[: n - criterion.n_lookback])
    improvement = hv_now - hv_before
    return bool(improvement < criterion.min_improvement)
