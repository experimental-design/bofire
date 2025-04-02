from typing import Annotated, List, Literal, Optional, Type

from pydantic import Field, model_validator

from bofire.data_models.base import BaseModel
from bofire.data_models.constraints.api import Constraint
from bofire.data_models.domain.api import Domain
from bofire.data_models.features.api import CategoricalInput, CategoricalOutput, Feature
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


def validate_domain_compatibility(domain1: Domain, domain2: Domain):
    """Validates if two domains are compatible to each other.

    To be compatible it is necessary that they have the same number
    of features, the same feature keys and that the
    features with the same key have the same type and categories.
    The bounds and allowed categories of the features can vary.

    Args:
        domain1 (Domain): First domain to be compared.
        domain2 (Domain): Second domain to be compared.

    Raises:
        ValueError: If one of the the conditions mentioned above is not met.
    """

    def validate(equals: List[str], features1, features2):
        for key in equals:
            feature1 = features1.get_by_key(key)
            feature2 = features2.get_by_key(key)
            if feature1.__class__ != feature2.__class__:
                raise ValueError(
                    f"Features with key {feature1.key} have different types."
                )
            if isinstance(
                feature1, (CategoricalInput, CategoricalOutput)
            ) and isinstance(feature2, (CategoricalInput, CategoricalOutput)):
                if feature1.categories != feature2.categories:
                    raise ValueError(
                        f"Features with key {feature1.key} have different categories."
                    )

    validate(
        [key for key in domain1.inputs.get_keys() if key in domain2.inputs.get_keys()],
        domain1.inputs,
        domain2.inputs,
    )
    validate(
        [
            key
            for key in domain1.outputs.get_keys()
            if key in domain2.outputs.get_keys()
        ],
        domain1.outputs,
        domain2.outputs,
    )


class StepwiseStrategy(Strategy):
    type: Literal["StepwiseStrategy"] = "StepwiseStrategy"  # type: ignore
    steps: Annotated[List[Step], Field(min_length=2)]

    @model_validator(mode="after")
    def validate_steps(self):
        for i, step in enumerate(self.steps):
            validate_domain_compatibility(self.domain, step.strategy_data.domain)
            if i < len(self.steps) - 1 and isinstance(
                step.condition, AlwaysTrueCondition
            ):
                raise ValueError(
                    "`AlwaysTrueCondition` is only allowed for the last step.",
                )
        return self

    @classmethod
    def is_feature_implemented(cls, my_type: Type[Feature]) -> bool:
        return True

    def is_constraint_implemented(self, my_type: Type[Constraint]) -> bool:
        return True
