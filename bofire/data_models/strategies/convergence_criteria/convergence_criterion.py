from typing import Any

from bofire.data_models.base import BaseModel


class ConvergenceCriterion(BaseModel):
    """Base class for all convergence criteria of a strategy.

    A convergence criterion answers the mathematical question whether the
    optimization has converged, i.e. whether we can still expect better
    experiments if we let the strategy continue. Simple budget-based stopping
    (e.g. a maximum number of experiments or steps) is intentionally not covered
    here, as it is already handled by the conditions of the ``StepwiseStrategy``.

    Convergence criteria only apply to ``PredictiveStrategy``s: they are an
    intrinsic property of a single model-based strategy and are consumed by the
    strategy itself via ``strategy.has_converged()``. The evaluator therefore has
    full access to the strategy's internals, including its fitted surrogate
    model(s), which it reads directly from the strategy when needed.

    Distinction from a stepwise ``Condition`` (see
    ``bofire.data_models.strategies.stepwise.conditions``):

    - A convergence criterion is an **intrinsic property of a single strategy**.
      It is consumed by the strategy itself via ``strategy.has_converged()`` and
      therefore has full access to the strategy's internals. This is the right
      home for model-aware stopping logic that can assess the progress of the
      optimization.
    - A stepwise ``Condition`` is **orchestration** between strategies: it selects
      the active step of a ``StepwiseStrategy`` and only sees the strategy through
      the minimal ``StepStrategy`` protocol (i.e. ``has_converged()`` alone, no
      model access). Convergence enters that layer solely through the
      ``StrategyHasConvergedCondition`` bridge.

    The data model only holds the parameters; the actual convergence logic is
    implemented in the functional layer (see
    ``bofire.strategies.convergence_criteria``).
    Custom criteria can be added via
    :func:`bofire.strategies.convergence_criteria.api.register`.
    """

    type: Any
