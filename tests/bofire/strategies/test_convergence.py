import pandas as pd
import pytest

import bofire.convergence_criteria.api as convergence_criteria
import bofire.data_models.convergence_criteria.api as convergence_data_models
import bofire.strategies.api as strategies
from bofire.benchmarks.single import Himmelblau
from bofire.data_models.convergence_criteria.api import (
    ObjectiveImprovementCriterion,
    ProposalDeviationCriterion,
)
from bofire.data_models.strategies.api import (
    RandomStrategy,
    StrategyHasConvergedCondition,
)


def _strategy_with_experiments(criterion, points, y):
    """Build a RandomStrategy on Himmelblau and tell it crafted experiments.

    Args:
        criterion: convergence criterion to attach to the strategy.
        points: list of ``(x_1, x_2)`` input locations.
        y: list of output values for the ``y`` output.
    """
    domain = Himmelblau().domain
    experiments = pd.DataFrame(points, columns=["x_1", "x_2"])
    experiments["y"] = y
    experiments["valid_y"] = 1
    strategy = strategies.map(
        RandomStrategy(domain=domain, convergence_criterion=criterion)
    )
    strategy.tell(experiments)
    return strategy


def test_convergence_criterion_serialization_roundtrip():
    for criterion in [
        ObjectiveImprovementCriterion(min_improvement=1e-2, n_lookback=5),
        ProposalDeviationCriterion(threshold=1e-3, n_consecutive=2),
    ]:
        reconstructed = type(criterion)(**criterion.model_dump())
        assert reconstructed == criterion


def test_convergence_criterion_requires_surrogate():
    assert (
        ObjectiveImprovementCriterion(
            min_improvement=1e-2, n_lookback=5
        ).requires_surrogate
        is False
    )
    assert ProposalDeviationCriterion(threshold=1e-3).requires_surrogate is False


def test_has_converged_without_convergence_criterion():
    domain = Himmelblau().domain
    strategy = strategies.map(RandomStrategy(domain=domain))
    assert strategy.has_converged() is False


# Himmelblau uses a MinimizeObjective with default bounds (0, 1), so the reward
# returned by the objectives is simply ``-y``: lower ``y`` means higher reward.


def test_objective_improvement_converged():
    # Best y stagnates at 1 over the lookback window -> no improvement.
    strategy = _strategy_with_experiments(
        ObjectiveImprovementCriterion(min_improvement=0.5, n_lookback=3),
        points=[(0, 0), (1, 1), (2, 2), (1, 1), (2, 2), (1, 1)],
        y=[10, 5, 1, 1, 1, 1],
    )
    assert strategy.has_converged() is True


def test_objective_improvement_not_converged():
    # Best y keeps dropping (10 -> 1) within the lookback window.
    strategy = _strategy_with_experiments(
        ObjectiveImprovementCriterion(min_improvement=0.5, n_lookback=3),
        points=[(0, 0), (1, 1), (2, 2), (1, 1), (2, 2), (1, 1)],
        y=[10, 9, 8, 5, 2, 1],
    )
    assert strategy.has_converged() is False


def test_objective_improvement_not_enough_experiments():
    strategy = _strategy_with_experiments(
        ObjectiveImprovementCriterion(min_improvement=0.5, n_lookback=3),
        points=[(0, 0), (1, 1), (2, 2)],
        y=[3, 2, 1],
    )
    assert strategy.has_converged() is False


def test_proposal_deviation_converged():
    # The last proposals coincide -> deviation is zero.
    strategy = _strategy_with_experiments(
        ProposalDeviationCriterion(threshold=1e-2, n_consecutive=2),
        points=[(0, 0), (3, 3), (1, 1), (1, 1), (1, 1)],
        y=[5, 4, 3, 3, 3],
    )
    assert strategy.has_converged() is True


def test_proposal_deviation_not_converged():
    # Proposals keep moving by a fixed step -> deviation stays above threshold.
    strategy = _strategy_with_experiments(
        ProposalDeviationCriterion(threshold=1e-2, n_consecutive=2),
        points=[(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)],
        y=[5, 4, 3, 2, 1],
    )
    assert strategy.has_converged() is False


def test_proposal_deviation_not_enough_experiments():
    strategy = _strategy_with_experiments(
        ProposalDeviationCriterion(threshold=1e-2, n_consecutive=2),
        points=[(0, 0), (1, 1)],
        y=[2, 1],
    )
    assert strategy.has_converged() is False


def test_has_converged_requires_missing_surrogate():
    from typing import Literal

    class _SurrogateRequiringCriterion(convergence_data_models.ConvergenceCriterion):
        type: Literal["_SurrogateRequiringCriterion"] = "_SurrogateRequiringCriterion"

        @property
        def requires_surrogate(self) -> bool:
            return True

    convergence_criteria.register(
        _SurrogateRequiringCriterion, lambda criterion, strategy, surrogates: True
    )

    domain = Himmelblau().domain
    strategy = strategies.map(
        RandomStrategy(
            domain=domain,
            convergence_criterion=_SurrogateRequiringCriterion(),
        )
    )
    with pytest.raises(ValueError, match="requires a surrogate model"):
        strategy.has_converged()


def test_map_unregistered_convergence_criterion():
    class _UnknownCriterion(convergence_data_models.ConvergenceCriterion):
        type: str = "_UnknownCriterion"

    with pytest.raises(KeyError, match="No convergence evaluator registered"):
        convergence_criteria.map(_UnknownCriterion())


def test_strategy_has_converged_condition_evaluate():
    condition = StrategyHasConvergedCondition()

    class _Strategy:
        def __init__(self, finished: bool):
            self._finished = finished

        def has_converged(self) -> bool:
            return self._finished

    # The step stays active (evaluate -> True) while the strategy has not converged.
    assert (
        condition.evaluate(_Strategy(finished=False), Himmelblau().domain, None) is True
    )
    assert (
        condition.evaluate(_Strategy(finished=True), Himmelblau().domain, None) is False
    )


def test_register_custom_convergence_criterion():
    from typing import Literal

    class _CustomConvergenceCriterion(convergence_data_models.ConvergenceCriterion):
        type: Literal["_CustomConvergenceCriterion"] = "_CustomConvergenceCriterion"

    calls = {}

    @convergence_criteria.register(_CustomConvergenceCriterion)
    def _evaluate(criterion, strategy, surrogates):
        calls["hit"] = True
        return True

    # After registration the custom criterion is accepted as a strategy field
    # and its evaluator is bound to the strategy and used by has_converged().
    domain = Himmelblau().domain
    strategy = strategies.map(
        RandomStrategy(
            domain=domain,
            convergence_criterion=_CustomConvergenceCriterion(),
        )
    )
    assert strategy.has_converged() is True
    assert calls["hit"] is True
