r"""Functional convergence evaluation for the proposal deviation criterion.

The evaluator is a pure function of the criterion and the strategy's *recorded
history*: it must not keep internal state between ``has_converged`` calls. The
realized proposals are read from the input locations of ``strategy.experiments``
(which accumulate and are never reset), so a strategy reconstructed by replaying
``tell`` reaches the same result.

Math:
    Let :math:`x_1, \dots, x_N` be the numeric (continuous and discrete) input
    locations of the recorded experiments and :math:`l_j, u_j` the bounds of
    numeric input :math:`j`. Each location is min-max normalized,

    .. math:: \tilde{x}_{ij} = (x_{ij} - l_j) / (u_j - l_j),

    the per-step deviation is the Euclidean distance between consecutive
    normalized proposals,

    .. math:: d_k = \lVert \tilde{x}_k - \tilde{x}_{k-1} \rVert_2,

    and convergence holds once the last ``n_consecutive`` deviations are all
    below ``threshold`` and no categorical input changed over those steps.
"""

from typing import TYPE_CHECKING

import numpy as np

from bofire.data_models.features.api import (
    CategoricalInput,
    ContinuousInput,
    DiscreteInput,
)
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
        below ``threshold`` and no categorical input changed over those steps,
        False otherwise (including when there are not yet enough experiments or
        no numeric and no categorical inputs).
    """
    experiments = strategy.experiments
    if experiments is None:
        return False

    # Continuous and discrete inputs are ordinal numerics with bounds; both are
    # min-max normalized and contribute to the Euclidean deviation.
    numeric_inputs = strategy.domain.inputs.get([ContinuousInput, DiscreteInput])
    categorical_inputs = strategy.domain.inputs.get(CategoricalInput)
    if len(numeric_inputs) == 0 and len(categorical_inputs) == 0:
        return False

    n = len(experiments)
    # Need n_consecutive + 1 proposals to form n_consecutive deviations.
    if n < criterion.n_consecutive + 1:
        return False

    # Min-max normalize the numeric input locations to the unit cube and take
    # the Euclidean distance between consecutive proposals.
    if len(numeric_inputs) > 0:
        normalized = np.empty((n, len(numeric_inputs)))
        for j, feat in enumerate(numeric_inputs):
            assert isinstance(feat, (ContinuousInput, DiscreteInput))
            lower, upper = feat.lower_bound, feat.upper_bound
            span = upper - lower
            column = experiments[feat.key].to_numpy(dtype=float)
            normalized[:, j] = (column - lower) / span if span > 0 else 0.0
        deviations = np.linalg.norm(np.diff(normalized, axis=0), axis=1)
    else:
        deviations = np.zeros(n - 1)

    recent_deviations = deviations[-criterion.n_consecutive :]
    if not np.all(recent_deviations < criterion.threshold):
        return False

    # A categorical proposal that differs from its predecessor counts as movement.
    if len(categorical_inputs) > 0:
        keys = [feat.key for feat in categorical_inputs]
        values = experiments[keys].to_numpy()
        recent = values[-(criterion.n_consecutive + 1) :]
        if np.any(recent[1:] != recent[:-1]):
            return False

    return True
