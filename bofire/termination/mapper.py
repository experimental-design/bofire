"""Mapper for termination conditions to their evaluators."""

from typing import Optional

from bofire.data_models.termination.api import (
    AnyTerminationCondition,
    CombiTerminationCondition,
    UCBLCBRegretTermination,
)
from bofire.termination.evaluator import TerminationEvaluator, UCBLCBRegretEvaluator


def map(
    termination_condition: AnyTerminationCondition,
) -> Optional[TerminationEvaluator]:
    """Map a termination condition to its corresponding evaluator.

    Not all termination conditions require an evaluator (e.g., MaxIterations),
    so this may return None.

    Args:
        termination_condition: The termination condition data model.

    Returns:
        The corresponding evaluator, or None if not needed.
    """
    if isinstance(termination_condition, UCBLCBRegretTermination):
        return UCBLCBRegretEvaluator()

    if isinstance(termination_condition, CombiTerminationCondition):
        # For combined conditions, return a list would be more appropriate
        # but for now we return None and handle in the runner
        return None

    # MaxIterationsTermination, AlwaysContinue don't need evaluators
    return None


def get_required_evaluators(
    termination_condition: AnyTerminationCondition,
) -> list:
    """Get all evaluators required for a termination condition.

    For CombiTerminationCondition, this returns evaluators for all sub-conditions.

    Args:
        termination_condition: The termination condition data model.

    Returns:
        List of required evaluators.
    """
    evaluators = []

    if isinstance(termination_condition, CombiTerminationCondition):
        for sub_condition in termination_condition.conditions:
            sub_evaluators = get_required_evaluators(sub_condition)
            evaluators.extend(sub_evaluators)
    else:
        evaluator = map(termination_condition)
        if evaluator is not None:
            evaluators.append(evaluator)

    return evaluators
