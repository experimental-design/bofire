from typing import Literal

from pydantic import PositiveFloat, PositiveInt

from bofire.data_models.strategies.convergence_criteria.convergence_criterion import (
    ConvergenceCriterion,
)


class HypervolumeImprovementCriterion(ConvergenceCriterion):
    r"""Convergence based on the improvement of the dominated hypervolume.

    This is the multi-objective analogue of the
    :class:`ObjectiveImprovementCriterion`. Instead of a scalar reward, the
    progress of the optimization is measured by the hypervolume dominated by the
    recorded experiments with respect to a reference point.

    Let :math:`H_k` be the hypervolume dominated by the Pareto front of the first
    :math:`k` recorded experiments (in chronological order) with respect to a
    common reference point inferred from all experiments. Because adding an
    experiment can never decrease the dominated hypervolume, :math:`H_k` is
    monotonically non-decreasing and already equals the best value observed
    within the first :math:`k` experiments.

    The optimization is considered converged once the dominated hypervolume has
    improved by less than ``min_improvement`` over the last ``n_lookback``
    experiments,

    .. math:: H_N - H_{N - \text{n\_lookback}} < \text{min\_improvement}.

    At least ``n_lookback + 1`` experiments with valid outputs are required;
    otherwise the strategy is not considered converged. The criterion requires at
    least two output features with an optimization objective.

    Attributes:
        min_improvement: Minimal improvement of the dominated hypervolume that is
            still considered relevant.
        n_lookback: Number of most recent experiments over which the improvement
            is evaluated.
    """

    type: Literal["HypervolumeImprovementCriterion"] = "HypervolumeImprovementCriterion"
    min_improvement: PositiveFloat
    n_lookback: PositiveInt

    @classmethod
    def is_applicable_to_multiobjective(cls) -> bool:
        return True

    @classmethod
    def is_applicable_to_singleobjective(cls) -> bool:
        return False

    @classmethod
    def is_applicable_to_objective_free(cls) -> bool:
        return False
