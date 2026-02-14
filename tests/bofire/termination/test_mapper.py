"""Tests for termination mapper."""

from bofire.data_models.termination.api import (
    AlwaysContinue,
    CombiTerminationCondition,
    MaxIterationsTermination,
    UCBLCBRegretTermination,
)
from bofire.termination.evaluator import UCBLCBRegretEvaluator
from bofire.termination.mapper import get_required_evaluators, map


class TestMap:
    def test_map_ucblcb(self):
        condition = UCBLCBRegretTermination()
        evaluator = map(condition)
        assert isinstance(evaluator, UCBLCBRegretEvaluator)

    def test_map_max_iterations_returns_none(self):
        condition = MaxIterationsTermination(max_iterations=100)
        evaluator = map(condition)
        assert evaluator is None

    def test_map_always_continue_returns_none(self):
        condition = AlwaysContinue()
        evaluator = map(condition)
        assert evaluator is None

    def test_map_combi_returns_none(self):
        condition = CombiTerminationCondition(
            conditions=[
                MaxIterationsTermination(max_iterations=100),
                UCBLCBRegretTermination(),
            ]
        )
        evaluator = map(condition)
        assert evaluator is None


class TestGetRequiredEvaluators:
    def test_single_condition_needing_evaluator(self):
        condition = UCBLCBRegretTermination()
        evaluators = get_required_evaluators(condition)
        assert len(evaluators) == 1
        assert isinstance(evaluators[0], UCBLCBRegretEvaluator)

    def test_single_condition_not_needing_evaluator(self):
        condition = MaxIterationsTermination(max_iterations=100)
        evaluators = get_required_evaluators(condition)
        assert len(evaluators) == 0

    def test_combi_condition(self):
        condition = CombiTerminationCondition(
            conditions=[
                MaxIterationsTermination(max_iterations=100),
                UCBLCBRegretTermination(),
            ]
        )
        evaluators = get_required_evaluators(condition)
        assert len(evaluators) == 1
        types = {type(e) for e in evaluators}
        assert UCBLCBRegretEvaluator in types

    def test_nested_combi_condition(self):
        inner = CombiTerminationCondition(
            conditions=[
                UCBLCBRegretTermination(),
                MaxIterationsTermination(max_iterations=100),
            ]
        )
        outer = CombiTerminationCondition(
            conditions=[inner, MaxIterationsTermination(max_iterations=50)],
        )
        evaluators = get_required_evaluators(outer)
        assert len(evaluators) == 1
