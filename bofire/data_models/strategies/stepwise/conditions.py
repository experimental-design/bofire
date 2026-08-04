from abc import abstractmethod
from typing import Annotated, Any, List, Literal, Optional, Protocol

import pandas as pd
from pydantic import Field, PositiveInt, field_validator

from bofire.data_models.base import BaseModel
from bofire.data_models.domain.api import Domain
from bofire.data_models.objectives.api import ConstrainedObjective
from bofire.data_models.unions import tagged_union


class StepStrategy(Protocol):
    """Minimal interface of the step strategy needed to evaluate conditions."""

    def has_converged(self) -> bool: ...


class EvaluateableCondition:
    """Mixin for stepwise conditions that select the active step.

    A condition is **orchestration** logic for a ``StepwiseStrategy``: it decides
    which step is currently active, based on the ``domain`` and ``experiments``
    and, at most, the minimal ``StepStrategy`` protocol (``has_converged()``).
    Conditions deliberately do **not** get access to a strategy's surrogate
    model(s); model-aware stopping logic belongs in a ``ConvergenceCriterion``
    (see ``bofire.data_models.strategies.convergence_criteria``), which runs inside the
    strategy and is surfaced here only through ``StrategyHasConvergedCondition``.

    Note the polarity: ``evaluate`` returns ``True`` while the step should *stay
    active* (i.e. it is **not** yet done), which is the opposite of
    ``strategy.has_converged()``. The ``StrategyHasConvergedCondition`` bridge
    reconciles the two by returning ``not strategy.has_converged()``.
    """

    @abstractmethod
    def evaluate(
        self,
        strategy: StepStrategy,
        domain: Domain,
        experiments: Optional[pd.DataFrame],
    ) -> bool:
        """Whether the step using ``strategy`` should remain the active step.

        This is the polymorphic hook used by the ``StepwiseStrategy`` to select
        the active step. Data-based conditions only look at ``domain`` and
        ``experiments`` and ignore ``strategy``; conditions that depend on the
        strategy itself (e.g. whether it is finished) inspect ``strategy``.
        """
        pass


class Condition(BaseModel):
    type: Any


class SingleCondition(BaseModel):
    type: Any


class FeasibleExperimentCondition(SingleCondition, EvaluateableCondition):
    """Condition to check if a certain number of feasible experiments are available.

    For this purpose, the condition checks if there are any kind of ConstrainedObjective's
    in the domain. If, yes it checks if there is a certain number of feasible experiments.
    The condition is fulfilled if the number of feasible experiments is smaller than
    the number of required feasible experiments. It is not fulfilled when there are no
    ConstrainedObjective's in the domain.
    This condition can be used in scenarios where there is a large amount of output constraints
    and one wants to make sure that they are fulfilled before optimizing the actual objective(s).
    To do this, it is best to combine this condition with the SoboStrategy and qLogPF
    as acquisition function.

    Attributes:
        n_required_feasible_experiments: Number of required feasible experiments.
        threshold: Threshold for the feasibility calculation. Default is 0.9.
    """

    type: Literal["FeasibleExperimentCondition"] = "FeasibleExperimentCondition"
    n_required_feasible_experiments: PositiveInt = 1
    threshold: Annotated[float, Field(ge=0, le=1)] = 0.9

    def evaluate(
        self,
        strategy: StepStrategy,
        domain: Domain,
        experiments: Optional[pd.DataFrame],
    ) -> bool:
        constrained_outputs = domain.outputs.get_by_objective(ConstrainedObjective)
        if len(constrained_outputs) == 0:
            return False

        if experiments is None:
            return True

        valid_experiments = (
            constrained_outputs.preprocess_experiments_all_valid_outputs(experiments)
        )

        valid_experiments = valid_experiments[domain.is_fulfilled(valid_experiments)]

        feasibilities = pd.concat(
            [
                feat(
                    valid_experiments[feat.key],
                    valid_experiments[feat.key],
                )
                for feat in constrained_outputs
            ],
            axis=1,
        ).product(axis=1)

        return bool(
            feasibilities[feasibilities >= self.threshold].sum()
            < self.n_required_feasible_experiments
        )


class NumberOfExperimentsCondition(SingleCondition, EvaluateableCondition):
    type: Literal["NumberOfExperimentsCondition"] = "NumberOfExperimentsCondition"
    n_experiments: Annotated[int, Field(ge=1)]

    def evaluate(
        self,
        strategy: StepStrategy,
        domain: Domain,
        experiments: Optional[pd.DataFrame],
    ) -> bool:
        if experiments is None:
            n_experiments = 0
        else:
            n_experiments = len(
                domain.outputs.preprocess_experiments_all_valid_outputs(experiments),
            )
        return n_experiments < self.n_experiments


class AlwaysTrueCondition(SingleCondition, EvaluateableCondition):
    type: Literal["AlwaysTrueCondition"] = "AlwaysTrueCondition"

    def evaluate(
        self,
        strategy: StepStrategy,
        domain: Domain,
        experiments: Optional[pd.DataFrame],
    ) -> bool:
        return True


class StrategyHasConvergedCondition(SingleCondition, EvaluateableCondition):
    """Condition that keeps the current step active until its strategy converged.

    In contrast to the other conditions, this condition cannot be evaluated from
    the domain and experiments alone. It asks the strategy of the current step
    whether it has converged, i.e. whether its convergence criterion is met.

    The current step stays active as long as its strategy has not converged. Once
    the strategy reports that it has converged, the ``StepwiseStrategy`` advances
    to the next step.
    """

    type: Literal["StrategyHasConvergedCondition"] = "StrategyHasConvergedCondition"

    def evaluate(
        self,
        strategy: StepStrategy,
        domain: Domain,
        experiments: Optional[pd.DataFrame],
    ) -> bool:
        return not strategy.has_converged()


class CombiCondition(Condition, EvaluateableCondition):
    type: Literal["CombiCondition"] = "CombiCondition"
    conditions: Annotated[
        List[
            tagged_union(
                NumberOfExperimentsCondition,
                "CombiCondition",
                AlwaysTrueCondition,
            )
        ],
        Field(min_length=2),
    ]
    n_required_conditions: Annotated[int, Field(ge=0)]

    @field_validator("n_required_conditions")
    @classmethod
    def validate_n_required_conditions(cls, v, info):
        if v > len(info.data["conditions"]):
            raise ValueError(
                "Number of required conditions larger than number of conditions.",
            )
        return v

    def evaluate(
        self,
        strategy: StepStrategy,
        domain: Domain,
        experiments: Optional[pd.DataFrame],
    ) -> bool:
        n_matched_conditions = 0
        for c in self.conditions:
            if c.evaluate(strategy, domain, experiments):
                n_matched_conditions += 1
        if n_matched_conditions >= self.n_required_conditions:
            return True
        return False


AnyCondition = tagged_union(
    NumberOfExperimentsCondition,
    CombiCondition,
    AlwaysTrueCondition,
    FeasibleExperimentCondition,
    StrategyHasConvergedCondition,
)
