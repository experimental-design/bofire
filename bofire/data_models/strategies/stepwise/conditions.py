from abc import abstractmethod
from typing import Annotated, List, Literal, Optional, Union

import pandas as pd
from pydantic import Field, field_validator

from bofire.data_models.base import BaseModel
from bofire.data_models.domain.api import Domain


class EvaluateableCondition:
    @abstractmethod
    def evaluate(self, domain: Domain, experiments: Optional[pd.DataFrame]) -> bool:
        pass


class Condition(BaseModel):
    type: str


class SingleCondition(BaseModel):
    type: str


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


AnyCondition = Union[NumberOfExperimentsCondition, CombiCondition, AlwaysTrueCondition]
