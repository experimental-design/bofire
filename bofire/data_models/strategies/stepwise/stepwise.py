from typing import List, Literal, Optional, Type

from pydantic import Field, field_validator
from typing_extensions import Annotated

from bofire.data_models.base import BaseModel
from bofire.data_models.constraints.api import Constraint
from bofire.data_models.features.api import Feature
from bofire.data_models.strategies.actual_strategy_type import ActualStrategy
from bofire.data_models.strategies.stepwise.conditions import (
    AlwaysTrueCondition,
    AnyCondition,
)
from bofire.data_models.strategies.strategy import Strategy
from bofire.data_models.transforms.api import AnyTransform


class Step(BaseModel):
    type: Literal["Step"] = "Step"
    strategy_data: ActualStrategy
    condition: AnyCondition
    transform: Optional[AnyTransform] = None


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
