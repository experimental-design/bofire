"""Mapping and registration of strategy convergence criteria.

This module wires convergence criterion data models (see
:mod:`bofire.data_models.strategies.convergence_criteria`) to their functional
evaluators. The evaluators themselves live in dedicated per-criterion modules
(e.g. :mod:`bofire.strategies.convergence_criteria.objective_improvement`); this
module only holds the registry and the ``map`` / ``register`` machinery.

Custom criteria can be added at runtime via :func:`register`, which registers
both the data model (so it is accepted in strategy ``convergence_criterion``
fields) and its evaluator.
"""

from typing import Callable, Optional, Type

import bofire.data_models.strategies.convergence_criteria.api as data_models
from bofire.data_models.strategies.convergence_criteria.api import ConvergenceCriterion
from bofire.strategies.convergence_criteria.objective_improvement import (
    evaluate_objective_improvement_criterion,
)
from bofire.strategies.convergence_criteria.proposal_deviation import (
    evaluate_proposal_deviation_criterion,
)


CONVERGENCE_MAP: dict[type[ConvergenceCriterion], Callable[..., bool]] = {
    data_models.ObjectiveImprovementCriterion: (
        evaluate_objective_improvement_criterion
    ),
    data_models.ProposalDeviationCriterion: (evaluate_proposal_deviation_criterion),
}


def register(
    data_model_cls: Type[ConvergenceCriterion],
    evaluator: Optional[Callable] = None,
):
    """Register a custom convergence criterion and its evaluator.

    Can be used as a decorator or as a direct function call::

        # Decorator form
        @register(MyConvergenceCriterion)
        def evaluate_my_criterion(criterion, strategy):
            return ...

        # Direct call form
        register(MyConvergenceCriterion, evaluate_my_criterion)

    Args:
        data_model_cls: The Pydantic convergence criterion data model class.
        evaluator: A callable that takes ``(criterion, strategy)`` and returns
            whether the strategy is considered converged. It must be a pure
            function of the criterion and the strategy's recorded history and
            must not keep internal state between calls. If the criterion needs
            the strategy's surrogate model(s), it reads them from the strategy.
            If not provided, returns a decorator.

    Returns:
        The evaluator function (unchanged) when used as a decorator, None
        otherwise.
    """

    def _register(fn: Callable) -> Callable:
        # Register with the data model union first so a discriminator conflict
        # is raised before the functional map is touched (no partial state).
        data_models.register_convergence_criterion(data_model_cls)
        CONVERGENCE_MAP[data_model_cls] = fn

        return fn

    if evaluator is not None:
        _register(evaluator)
        return None

    return _register


def map(criterion: ConvergenceCriterion) -> Callable[..., bool]:
    """Map a convergence criterion data model to its evaluator function.

    This is used at strategy creation time to bind the evaluator to the
    strategy, so the strategy does not need to dispatch on the criterion type
    when checking for convergence. The returned evaluator is called with
    ``(criterion, strategy)`` and returns whether the strategy is considered
    converged.

    Args:
        criterion: The convergence criterion data model to map.

    Returns:
        Callable[..., bool]: The evaluator function for the criterion.

    Raises:
        KeyError: If no evaluator is registered for the criterion type.
    """
    try:
        return CONVERGENCE_MAP[type(criterion)]
    except KeyError:
        raise KeyError(
            f"No convergence evaluator registered for `{type(criterion).__name__}`.",
        )
