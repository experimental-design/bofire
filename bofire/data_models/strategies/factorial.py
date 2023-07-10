from typing import Literal, Type

from bofire.data_models.features.api import (
    CategoricalDescriptorInput,
    CategoricalInput,
    ContinuousOutput,
    DiscreteInput,
    Feature,
)
from bofire.data_models.strategies.strategy import Strategy


class FactorialStrategy(Strategy):
    type: Literal["FactorialStrategy"] = "FactorialStrategy"

    @classmethod
    def is_constraint_implemented(cls, my_type: Type[Feature]) -> bool:
        return False

    @classmethod
    def is_feature_implemented(cls, my_type: Type[Feature]) -> bool:
        return my_type in [
            CategoricalInput,
            DiscreteInput,
            CategoricalDescriptorInput,
            ContinuousOutput,
        ]
