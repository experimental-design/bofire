import warnings
from typing import Literal, Type

from pydantic import model_validator

from bofire.data_models.constraints.api import Constraint
from bofire.data_models.features.api import (
    CategoricalDescriptorInput,
    CategoricalInput,
    ContinuousOutput,
    DiscreteInput,
    Feature,
)
from bofire.data_models.strategies.strategy import Strategy


class FactorialStrategy(Strategy):
    """Factorial design strategy.

    This strategy is deprecated, please use FractionalFactorialStrategy instead.
    """

    type: Literal["FactorialStrategy"] = "FactorialStrategy"  # type: ignore

    def is_constraint_implemented(self, my_type: Type[Constraint]) -> bool:
        return False

    @classmethod
    def is_feature_implemented(cls, my_type: Type[Feature]) -> bool:
        return my_type in [
            CategoricalInput,
            DiscreteInput,
            CategoricalDescriptorInput,
            ContinuousOutput,
        ]

    @model_validator(mode="after")
    def raise_depreaction_warning(self):
        warnings.warn(
            "`FactorialStrategy is deprecated, use `FractionalFactorialStrategy` instead.",
            DeprecationWarning,
        )
        return self
