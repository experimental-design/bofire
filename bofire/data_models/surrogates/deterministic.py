from typing import Annotated, Dict, Literal, Type

from pydantic import Field, model_validator

from bofire.data_models.features.api import (
    AnyOutput,
    ContinuousInput,
    ContinuousOutput,
    DiscreteInput,
)
from bofire.data_models.surrogates.botorch import BotorchSurrogate


class LinearDeterministicSurrogate(BotorchSurrogate):
    type: Literal["LinearDeterministicSurrogate"] = "LinearDeterministicSurrogate"
    coefficients: Annotated[Dict[str, float], Field(min_length=1)]
    intercept: float

    @classmethod
    def is_output_implemented(cls, my_type: Type[AnyOutput]) -> bool:
        """Abstract method to check output type for surrogate models
        Args:
            my_type: continuous or categorical output
        Returns:
            bool: True if the output type is valid for the surrogate chosen, False otherwise
        """
        return isinstance(my_type, type(ContinuousOutput))

    @model_validator(mode="after")
    def validate_input_types(self):
        if len(self.inputs.get([ContinuousInput, DiscreteInput])) != len(self.inputs):
            raise ValueError(
                "Only numerical inputs are suppoerted for the `LinearDeterministicSurrogate`"
            )
        return self

    @model_validator(mode="after")
    def validate_coefficients(self):
        if sorted(self.inputs.get_keys()) != sorted(self.coefficients.keys()):
            raise ValueError("coefficient keys do not match input feature keys.")
        return self
