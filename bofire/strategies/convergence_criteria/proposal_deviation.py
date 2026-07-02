r"""Functional convergence evaluation for the proposal deviation criterion.

The evaluator is a pure function of the criterion and the strategy's *recorded
history*: it must not keep internal state between ``has_converged`` calls. The
realized proposals are read from the input locations of ``strategy.experiments``
(which accumulate and are never reset), so a strategy reconstructed by replaying
``tell`` reaches the same result.

Math:
    Let :math:`x_1, \dots, x_N` be the continuous input locations of the recorded
    experiments and :math:`l_j, u_j` the bounds of continuous input :math:`j`.
    Each location is min-max normalized,

    .. math:: \tilde{x}_{ij} = (x_{ij} - l_j) / (u_j - l_j),

    the per-step deviation is the Euclidean distance between consecutive
    normalized proposals,

    .. math:: d_k = \lVert \tilde{x}_k - \tilde{x}_{k-1} \rVert_2,

    and convergence holds once the last ``n_consecutive`` deviations are all
    below ``threshold``.
"""

from typing import TYPE_CHECKING

import numpy as np

from bofire.data_models.features.api import ContinuousInput
from bofire.data_models.strategies.convergence_criteria.api import (
    ProposalDeviationCriterion,
)


if TYPE_CHECKING:
    from bofire.strategies.predictives.predictive import PredictiveStrategy


def evaluate_proposal_deviation_criterion(
    criterion: ProposalDeviationCriterion,
    strategy: "PredictiveStrategy",
) -> bool:
    """Evaluate whether consecutive proposals stopped moving.

    Args:
        criterion: The convergence criterion data model with its parameters.
        strategy: The functional strategy providing the recorded experiments.

    Returns:
        bool: True if the last ``n_consecutive`` normalized deviations are all
        below ``threshold``, False otherwise (including when there are not yet
        enough experiments or no continuous inputs).
    """
    experiments = strategy.experiments
    if experiments is None:
        return False

    continuous_inputs = strategy.domain.inputs.get(ContinuousInput)
    if len(continuous_inputs) == 0:
        return False

    n = len(experiments)
    # Need n_consecutive + 1 proposals to form n_consecutive deviations.
    if n < criterion.n_consecutive + 1:
        return False

    # Min-max normalize the continuous input locations to the unit cube.
    normalized = np.empty((n, len(continuous_inputs)))
    for j, feat in enumerate(continuous_inputs):
        assert isinstance(feat, ContinuousInput)
        lower, upper = feat.lower_bound, feat.upper_bound
        span = upper - lower
        column = experiments[feat.key].to_numpy(dtype=float)
        normalized[:, j] = (column - lower) / span if span > 0 else 0.0

    # Euclidean distance between consecutive proposals.
    deviations = np.linalg.norm(np.diff(normalized, axis=0), axis=1)

    recent = deviations[-criterion.n_consecutive :]
    return bool(np.all(recent < criterion.threshold))
