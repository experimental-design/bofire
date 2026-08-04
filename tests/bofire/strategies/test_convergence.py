import pandas as pd
import pytest

import bofire.strategies.api as strategies
import bofire.strategies.convergence_criteria.api as convergence_criteria
from bofire.benchmarks.single import Himmelblau
from bofire.data_models.domain.api import Domain, Inputs, Outputs
from bofire.data_models.features.api import ContinuousInput, ContinuousOutput
from bofire.data_models.objectives.api import MaximizeObjective
from bofire.data_models.strategies.api import (
    ActiveLearningStrategy,
    MoboStrategy,
    SoboStrategy,
    StrategyHasConvergedCondition,
)
from bofire.data_models.strategies.convergence_criteria.api import (
    ConvergenceCriterion as convergence_data_models_ConvergenceCriterion,
)
from bofire.data_models.strategies.convergence_criteria.api import (
    HypervolumeImprovementCriterion,
    ObjectiveImprovementCriterion,
    ProposalDeviationCriterion,
)


@pytest.fixture
def restore_convergence_registry():
    """Snapshot and restore the global convergence-criterion registries.

    Registering a custom criterion mutates process-global state (the
    ``AnyConvergenceCriterion`` union, the ``convergence_criterion`` field
    patched onto every predictive strategy, and the functional evaluator map).
    Restoring it after the test keeps the registrations from leaking into other
    tests.
    """
    import bofire.data_models.strategies.convergence_criteria.api as cc_data_api
    from bofire.data_models.strategies.convergence_criteria._register import (
        _rebuild_dependent_models,
    )
    from bofire.strategies.convergence_criteria.mapper import CONVERGENCE_MAP

    saved_types = list(cc_data_api._CONVERGENCE_CRITERION_TYPES)
    saved_union = cc_data_api.AnyConvergenceCriterion
    saved_map = dict(CONVERGENCE_MAP)
    try:
        yield
    finally:
        cc_data_api._CONVERGENCE_CRITERION_TYPES[:] = saved_types
        cc_data_api.AnyConvergenceCriterion = saved_union
        CONVERGENCE_MAP.clear()
        CONVERGENCE_MAP.update(saved_map)
        _rebuild_dependent_models()


def _strategy_with_experiments(criterion, points, y):
    """Build a SoboStrategy on Himmelblau and tell it crafted experiments.

    Convergence criteria only apply to ``PredictiveStrategy``s. The built-in
    criteria only read the recorded experiments, so the surrogate model is not
    fitted here (``retrain=False``) to keep the tests fast.

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
        SoboStrategy(domain=domain, convergence_criterion=criterion)
    )
    strategy.tell(experiments, retrain=False)
    return strategy


def _mobo_strategy_with_experiments(criterion, points, y1, y2):
    """Build a MoboStrategy on a 2-objective domain and tell it experiments.

    Both outputs use a ``MaximizeObjective`` and the reference point is inferred
    from the recorded experiments. The surrogate model is not fitted here
    (``retrain=False``) to keep the tests fast, as the built-in criteria only
    read the recorded experiments.

    Args:
        criterion: convergence criterion to attach to the strategy.
        points: list of ``(x_1, x_2)`` input locations.
        y1: list of output values for the ``y1`` output.
        y2: list of output values for the ``y2`` output.
    """
    domain = Domain(
        inputs=Inputs(
            features=[
                ContinuousInput(key="x_1", bounds=(0, 5)),
                ContinuousInput(key="x_2", bounds=(0, 5)),
            ]
        ),
        outputs=Outputs(
            features=[
                ContinuousOutput(key="y1", objective=MaximizeObjective()),
                ContinuousOutput(key="y2", objective=MaximizeObjective()),
            ]
        ),
    )
    experiments = pd.DataFrame(points, columns=["x_1", "x_2"])
    experiments["y1"] = y1
    experiments["y2"] = y2
    experiments["valid_y1"] = 1
    experiments["valid_y2"] = 1
    strategy = strategies.map(
        MoboStrategy(domain=domain, convergence_criterion=criterion)
    )
    strategy.tell(experiments, retrain=False)
    return strategy


def test_has_converged_without_convergence_criterion():
    domain = Himmelblau().domain
    strategy = strategies.map(SoboStrategy(domain=domain))
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


def test_hypervolume_improvement_converged():
    # The Pareto front is fully established by the first three experiments; the
    # last experiments are dominated, so the hypervolume does not grow.
    strategy = _mobo_strategy_with_experiments(
        HypervolumeImprovementCriterion(min_improvement=0.5, n_lookback=3),
        points=[(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (1, 1)],
        y1=[1, 2, 3, 1, 1, 1],
        y2=[1, 3, 2, 1, 1, 1],
    )
    assert strategy.has_converged() is True


def test_hypervolume_improvement_not_converged():
    # The Pareto front keeps expanding within the lookback window, so the
    # dominated hypervolume grows substantially.
    strategy = _mobo_strategy_with_experiments(
        HypervolumeImprovementCriterion(min_improvement=0.5, n_lookback=3),
        points=[(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (1, 2)],
        y1=[1.0, 1.1, 1.2, 2.0, 3.0, 4.0],
        y2=[1.0, 1.1, 1.2, 2.0, 3.0, 4.0],
    )
    assert strategy.has_converged() is False


def test_hypervolume_improvement_not_enough_experiments():
    strategy = _mobo_strategy_with_experiments(
        HypervolumeImprovementCriterion(min_improvement=0.5, n_lookback=3),
        points=[(0, 0), (1, 1), (2, 2)],
        y1=[1, 2, 3],
        y2=[3, 2, 1],
    )
    assert strategy.has_converged() is False


def test_hypervolume_improvement_single_objective():
    # A single-objective strategy does not support the hypervolume criterion, so
    # attaching it is rejected at construction time.
    domain = Himmelblau().domain
    with pytest.raises(
        ValueError,
        match="is not implemented for strategy",
    ):
        SoboStrategy(
            domain=domain,
            convergence_criterion=HypervolumeImprovementCriterion(
                min_improvement=0.5, n_lookback=3
            ),
        )


def test_criterion_objective_free_applicability():
    # Only the objective-agnostic proposal deviation criterion can be evaluated
    # without any objective; the objective-progress criteria cannot.
    assert ProposalDeviationCriterion.is_applicable_to_objective_free() is True
    assert ObjectiveImprovementCriterion.is_applicable_to_objective_free() is False
    assert HypervolumeImprovementCriterion.is_applicable_to_objective_free() is False


def _active_learning_domain() -> Domain:
    """A multi-output domain valid for the ActiveLearningStrategy."""
    return Domain(
        inputs=Inputs(
            features=[
                ContinuousInput(key="x_1", bounds=(0, 1)),
                ContinuousInput(key="x_2", bounds=(0, 1)),
            ]
        ),
        outputs=Outputs(
            features=[
                ContinuousOutput(key="y1", objective=MaximizeObjective()),
                ContinuousOutput(key="y2", objective=MaximizeObjective()),
            ]
        ),
    )


def test_active_learning_accepts_proposal_deviation_criterion():
    # Active learning does not optimize an objective but still produces
    # proposals, so the objective-agnostic proposal deviation criterion applies.
    strategy = ActiveLearningStrategy(
        domain=_active_learning_domain(),
        convergence_criterion=ProposalDeviationCriterion(
            threshold=1e-2, n_consecutive=2
        ),
    )
    assert isinstance(strategy.convergence_criterion, ProposalDeviationCriterion)


@pytest.mark.parametrize(
    "criterion",
    [
        ObjectiveImprovementCriterion(min_improvement=0.5, n_lookback=3),
        HypervolumeImprovementCriterion(min_improvement=0.5, n_lookback=3),
    ],
)
def test_active_learning_rejects_objective_based_criteria(criterion):
    # Objective-progress criteria require objective information, which active
    # learning does not optimize, so attaching them is rejected.
    with pytest.raises(ValueError, match="is not implemented for strategy"):
        ActiveLearningStrategy(
            domain=_active_learning_domain(),
            convergence_criterion=criterion,
        )


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


def test_has_converged_requires_missing_surrogate(restore_convergence_registry):
    from typing import Literal

    class _SurrogateRequiringCriterion(convergence_data_models_ConvergenceCriterion):
        type: Literal["_SurrogateRequiringCriterion"] = "_SurrogateRequiringCriterion"

        @classmethod
        def is_applicable_to_singleobjective(cls) -> bool:
            return True

        @classmethod
        def is_applicable_to_multiobjective(cls) -> bool:
            return False

        @classmethod
        def is_applicable_to_objective_free(cls) -> bool:
            return False

    def _evaluate(criterion, strategy):
        # A custom criterion may access the strategy's surrogate model(s)
        # directly; here we simply assert that a fitted strategy exposes them.
        assert strategy.surrogates is not None
        return True

    convergence_criteria.register(_SurrogateRequiringCriterion, _evaluate)

    benchmark = Himmelblau()
    experiments = benchmark.f(benchmark.domain.inputs.sample(5), return_complete=True)
    strategy = strategies.map(
        SoboStrategy(
            domain=benchmark.domain,
            convergence_criterion=_SurrogateRequiringCriterion(),
        )
    )
    strategy.tell(experiments)
    assert strategy.has_converged() is True


def test_map_unregistered_convergence_criterion():
    class _UnknownCriterion(convergence_data_models_ConvergenceCriterion):
        type: str = "_UnknownCriterion"

        @classmethod
        def is_applicable_to_singleobjective(cls) -> bool:
            return True

        @classmethod
        def is_applicable_to_multiobjective(cls) -> bool:
            return True

        @classmethod
        def is_applicable_to_objective_free(cls) -> bool:
            return True

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


def test_register_custom_convergence_criterion(restore_convergence_registry):
    from typing import Literal

    class _CustomConvergenceCriterion(convergence_data_models_ConvergenceCriterion):
        type: Literal["_CustomConvergenceCriterion"] = "_CustomConvergenceCriterion"

        @classmethod
        def is_applicable_to_singleobjective(cls) -> bool:
            return True

        @classmethod
        def is_applicable_to_multiobjective(cls) -> bool:
            return False

        @classmethod
        def is_applicable_to_objective_free(cls) -> bool:
            return False

    calls = {}

    @convergence_criteria.register(_CustomConvergenceCriterion)
    def _evaluate(criterion, strategy):
        calls["hit"] = True
        return True

    # After registration the custom criterion is accepted as a strategy field
    # and its evaluator is bound to the strategy and used by has_converged().
    domain = Himmelblau().domain
    strategy = strategies.map(
        SoboStrategy(
            domain=domain,
            convergence_criterion=_CustomConvergenceCriterion(),
        )
    )
    assert strategy.has_converged() is True
    assert calls["hit"] is True
