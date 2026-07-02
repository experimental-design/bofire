from typing import Literal

from pydantic import PositiveFloat, PositiveInt

from bofire.data_models.strategies.convergence_criteria.convergence_criterion import (
    ConvergenceCriterion,
)


class ProposalDeviationCriterion(ConvergenceCriterion):
    r"""Convergence based on the deviation between consecutive proposals.

    This mirrors the classical notion of convergence of an iterative optimizer:
    the algorithm has converged once its iterates stop moving. Here the iterates
    are the input locations of the recorded experiments (i.e. the realized
    proposals), taken in chronological order.

    Let :math:`x_1, \dots, x_N` be the continuous input locations of the recorded
    experiments and let :math:`l_j, u_j` be the lower/upper bound of continuous
    input :math:`j`. Every location is min-max normalized to the unit cube,

    .. math:: \tilde{x}_{ij} = \frac{x_{ij} - l_j}{u_j - l_j},

    and the deviation of step :math:`k` is the Euclidean distance between two
    consecutive normalized proposals,

    .. math:: d_k = \lVert \tilde{x}_k - \tilde{x}_{k-1} \rVert_2 .

    The optimization is considered converged once the last ``n_consecutive``
    deviations all stay below ``threshold``,

    .. math:: d_k < \text{threshold} \quad
        \forall\, k \in \{N - \text{n\_consecutive} + 1, \dots, N\} .

    Only continuous inputs enter the distance; this is a deliberately simple
    check that does not rely on any surrogate model.

    Attributes:
        threshold: Normalized distance below which two consecutive proposals are
            considered to coincide.
        n_consecutive: Number of consecutive deviations that have to stay below
            the threshold before the strategy is considered converged.
    """

    type: Literal["ProposalDeviationCriterion"] = "ProposalDeviationCriterion"
    threshold: PositiveFloat
    n_consecutive: PositiveInt = 1
