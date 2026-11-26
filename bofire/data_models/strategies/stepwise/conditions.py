from abc import abstractmethod
from typing import Annotated, Any, Literal, Protocol

import numpy as np
import pandas as pd
from pydantic import Field, PositiveFloat, PositiveInt, field_validator, model_validator

from bofire.data_models.base import BaseModel
from bofire.data_models.domain.api import Domain
from bofire.data_models.features.api import CategoricalInput, CategoricalOutput
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
        experiments: pd.DataFrame | None,
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
        experiments: pd.DataFrame | None,
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
        experiments: pd.DataFrame | None,
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
        experiments: pd.DataFrame | None,
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
        experiments: pd.DataFrame | None,
    ) -> bool:
        return not strategy.has_converged()


def _n_distinguishable(values: np.ndarray, threshold: float) -> int:
    """Size of the largest subset of `values` whose elements are pairwise at
    least `threshold` apart; NaNs are ignored.

    In 1D, greedily keeping the smallest value that is at least `threshold`
    above the last kept one is exact.
    """
    values = np.sort(values[~np.isnan(values)])
    count, last = 0, -np.inf
    for v in values:
        if v - last >= threshold:
            count, last = count + 1, v
    return count


class DistinguishableParameters(BaseModel):
    """How many distinguishable values one feature needs to show.

    Examples:
        >>> DistinguishableParameters(
        ...     key="temperature", threshold=5.0, n_different_elements=4
        ... )
    """

    key: str = Field(description="Key of the feature whose values are checked.")
    threshold: PositiveFloat = Field(
        description=(
            "Minimal absolute difference for two values to count as different. "
            "Ignored for categorical features, whose values are different if "
            "they are different categories."
        ),
    )
    n_different_elements: PositiveInt = Field(
        description="Required number of pairwise different values.",
    )


class SufficientDistinguishableValues(SingleCondition, EvaluateableCondition):
    """Keeps the step active until features show enough distinguishable values.

    For every listed feature, the largest number of its observed values that are
    pairwise at least the feature's threshold apart is determined. The step stays
    active as long as this number is below the required count for at least one
    feature, e.g., to keep exploring until the inputs are spread out or a model
    has something other than noise around a single level to learn from.

    Numerical and categorical inputs as well as continuous and categorical
    outputs can be listed. For outputs, only valid values are used; repeated
    measurements of a continuous output at the same input point are replaced by
    their median first, so that replicates count only once and a single outlier
    among them does not shift the value.

    Examples:
        >>> SufficientDistinguishableValues(
        ...     inputs=[
        ...         DistinguishableParameters(
        ...             key="temperature", threshold=5.0, n_different_elements=4
        ...         )
        ...     ],
        ...     outputs=[
        ...         DistinguishableParameters(
        ...             key="yield", threshold=0.5, n_different_elements=3
        ...         )
        ...     ],
        ... )
    """

    type: Literal["SufficientDistinguishableValues"] = "SufficientDistinguishableValues"
    inputs: list[DistinguishableParameters] = Field(
        default=[], description="Inputs whose values are checked."
    )
    outputs: list[DistinguishableParameters] = Field(
        default=[], description="Outputs whose values are checked."
    )

    @model_validator(mode="after")
    def validate_not_empty(self):
        if not self.inputs and not self.outputs:
            raise ValueError("At least one input or output must be listed.")
        return self

    def evaluate(
        self,
        strategy: StepStrategy,
        domain: Domain,
        experiments: pd.DataFrame | None,
    ) -> bool:
        if experiments is None:
            return True
        for params in self.inputs:
            values = experiments[params.key]
            if isinstance(domain.inputs.get_by_key(params.key), CategoricalInput):
                n_distinguishable = values.nunique()
            else:
                n_distinguishable = _n_distinguishable(
                    values.to_numpy(dtype=float), params.threshold
                )
            if n_distinguishable < params.n_different_elements:
                return True
        input_keys = domain.inputs.get_keys()
        for params in self.outputs:
            valid = domain.outputs.preprocess_experiments_one_valid_output(
                params.key, experiments
            )
            if isinstance(domain.outputs.get_by_key(params.key), CategoricalOutput):
                n_distinguishable = valid[params.key].nunique()
            else:
                values = valid.groupby(input_keys, sort=False)[params.key].median()
                n_distinguishable = _n_distinguishable(
                    values.to_numpy(dtype=float), params.threshold
                )
            if n_distinguishable < params.n_different_elements:
                return True
        return False


class CombiCondition(Condition, EvaluateableCondition):
    type: Literal["CombiCondition"] = "CombiCondition"
    conditions: Annotated[
        list[
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
        experiments: pd.DataFrame | None,
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
    SufficientDistinguishableValues,
)
