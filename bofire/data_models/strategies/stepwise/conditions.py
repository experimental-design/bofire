from abc import abstractmethod
from typing import Annotated, Any, List, Literal, Optional, Union

import pandas as pd
from pydantic import Field, PositiveInt, field_validator

from bofire.data_models.base import BaseModel
from bofire.data_models.domain.api import Domain
from bofire.data_models.objectives.api import ConstrainedObjective


class EvaluateableCondition:
    @abstractmethod
    def evaluate(self, domain: Domain, experiments: Optional[pd.DataFrame]) -> bool:
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

    def evaluate(self, domain: Domain, experiments: Optional[pd.DataFrame]) -> bool:
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
                    valid_experiments[feat.key],  # type: ignore
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

    def evaluate(self, domain: Domain, experiments: Optional[pd.DataFrame]) -> bool:
        if experiments is None:
            n_experiments = 0
        else:
            n_experiments = len(
                domain.outputs.preprocess_experiments_all_valid_outputs(experiments),
            )
        return n_experiments < self.n_experiments


class AlwaysTrueCondition(SingleCondition, EvaluateableCondition):
    type: Literal["AlwaysTrueCondition"] = "AlwaysTrueCondition"

    def evaluate(self, domain: Domain, experiments: Optional[pd.DataFrame]) -> bool:
        return True


class CombiCondition(Condition, EvaluateableCondition):
    type: Literal["CombiCondition"] = "CombiCondition"
    conditions: Annotated[
        List[
            Union[NumberOfExperimentsCondition, "CombiCondition", AlwaysTrueCondition]
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

    def evaluate(self, domain: Domain, experiments: Optional[pd.DataFrame]) -> bool:
        n_matched_conditions = 0
        for c in self.conditions:
            if c.evaluate(domain, experiments):
                n_matched_conditions += 1
        if n_matched_conditions >= self.n_required_conditions:
            return True
        return False


AnyCondition = Union[
    NumberOfExperimentsCondition,
    CombiCondition,
    AlwaysTrueCondition,
    FeasibleExperimentCondition,
]
