from typing import List, Literal, Type, Union

from pydantic import Field, field_validator
from typing_extensions import Annotated

from bofire.data_models.base import BaseModel
from bofire.data_models.constraints.api import Constraint
from bofire.data_models.features.api import Feature
from bofire.data_models.strategies.doe import DoEStrategy
from bofire.data_models.strategies.factorial import FactorialStrategy
from bofire.data_models.strategies.predictives.mobo import MoboStrategy
from bofire.data_models.strategies.predictives.qehvi import QehviStrategy
from bofire.data_models.strategies.predictives.qnehvi import QnehviStrategy
from bofire.data_models.strategies.predictives.qparego import QparegoStrategy
from bofire.data_models.strategies.predictives.sobo import (
    AdditiveSoboStrategy,
    CustomSoboStrategy,
    MultiplicativeSoboStrategy,
    SoboStrategy,
)
from bofire.data_models.strategies.random import RandomStrategy
from bofire.data_models.strategies.shortest_path import ShortestPathStrategy
from bofire.data_models.strategies.space_filling import SpaceFillingStrategy
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
    CustomSoboStrategy,
    QehviStrategy,
    QnehviStrategy,
    QparegoStrategy,
    SpaceFillingStrategy,
    RandomStrategy,
    DoEStrategy,
    FactorialStrategy,
    MoboStrategy,
    ShortestPathStrategy,
]

AnyCondition = Union[NumberOfExperimentsCondition, CombiCondition, AlwaysTrueCondition]


class Step(BaseModel):
    type: Literal["Step"] = "Step"
    strategy_data: AnyStrategy
    condition: AnyCondition
    max_parallelism: Annotated[int, Field(ge=-1)]  # -1 means no restriction at all


class StepwiseStrategy(Strategy):
    type: Literal["StepwiseStrategy"] = "StepwiseStrategy"
    steps: Annotated[List[Step], Field(min_length=2)]

    @field_validator("steps")
    @classmethod
    def validate_steps(cls, v: List[Step], info):
        for i, step in enumerate(v):
            if step.strategy_data.domain != info.data["domain"]:
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
