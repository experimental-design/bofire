from typing import List, Literal, Type, Union

from pydantic import Field, validator
from typing_extensions import Annotated

from bofire.data_models.base import BaseModel
from bofire.data_models.constraints.api import Constraint
from bofire.data_models.features.api import Feature
from bofire.data_models.strategies.doe import DoEStrategy
from bofire.data_models.strategies.predictives.qehvi import QehviStrategy
from bofire.data_models.strategies.predictives.qnehvi import QnehviStrategy
from bofire.data_models.strategies.predictives.qparego import QparegoStrategy
from bofire.data_models.strategies.predictives.sobo import (
    AdditiveSoboStrategy,
    MultiplicativeSoboStrategy,
    SoboStrategy,
)
from bofire.data_models.strategies.random import RandomStrategy
from bofire.data_models.strategies.samplers.polytope import PolytopeSampler
from bofire.data_models.strategies.samplers.rejection import RejectionSampler
from bofire.data_models.strategies.stepwise.conditions import (
    AlwaysTrueCondition,
    CombiCondition,
    NumberOfExperimentsCondition,
)
from bofire.data_models.strategies.strategy import Strategy

AnyStrategy = Union[
    SoboStrategy,
    AdditiveSoboStrategy,
    MultiplicativeSoboStrategy,
    QehviStrategy,
    QnehviStrategy,
    QparegoStrategy,
    PolytopeSampler,
    RejectionSampler,
    RandomStrategy,
    DoEStrategy,
]

AnyCondition = Union[NumberOfExperimentsCondition, CombiCondition, AlwaysTrueCondition]


class Step(BaseModel):
    type: Literal["Step"] = "Step"
    strategy_data: AnyStrategy
    condition: AnyCondition
    max_parallelism: Annotated[int, Field(ge=-1)]  # -1 means no restriction at all


class StepwiseStrategy(Strategy):
    type: Literal["StepwiseStrategy"] = "StepwiseStrategy"
    steps: Annotated[List[Step], Field(min_items=2)]

    @validator("steps")
    def validate_steps(cls, v: List[Step], values):
        for i, step in enumerate(v):
            if step.strategy_data.domain != values["domain"]:
                raise ValueError(
                    f"Domain of step {i} is incompatible to domain of StepwiseStrategy."
                )
            if i < len(v) - 1 and isinstance(step.condition, AlwaysTrueCondition):
                raise ValueError(
                    "`AlwaysTrueCondition` is only allowed for the last step."
                )
        return v

    @classmethod
    def is_feature_implemented(cls, my_type: Type[Feature]) -> bool:
        return True

    @classmethod
    def is_constraint_implemented(cls, my_type: Type[Constraint]) -> bool:
        return True
