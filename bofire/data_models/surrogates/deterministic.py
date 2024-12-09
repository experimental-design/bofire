from typing import Annotated, Dict, Literal, Type

from pydantic import Field, model_validator

from bofire.data_models.features.api import (
    AnyOutput,
    CategoricalInput,
    ContinuousInput,
    ContinuousOutput,
    DiscreteInput,
)
from bofire.data_models.surrogates.botorch import BotorchSurrogate


class CategoricalDeterministicSurrogate(BotorchSurrogate):
    """Surrogate model that can be used to map categories of a categorical
    input feature to a deterministic continuous value.

    This is useful if one wants to penalize certain categories of an input feature
    more than others during the optimization process.

    Attributes:
        mapping: A dictionary mapping categories to deterministic float values.
    """

    type: Literal["CategoricalDeterministicSurrogate"] = (
        "CategoricalDeterministicSurrogate"
    )
    mapping: Annotated[Dict[str, float], Field(min_length=2)]

    @model_validator(mode="after")
    def validate_input_types(self):
        if len(self.inputs.get([CategoricalInput])) != len(self.inputs):
            raise ValueError(
                "Only categorical are supported for the `CategoricalDeterministicSurrogate`",
            )
        return self

    @classmethod
    def is_output_implemented(cls, my_type: Type[AnyOutput]) -> bool:
        """Checks output type for surrogate models

        Args:
            my_type: continuous or categorical output
        Returns:
            bool: True if the output type is valid for the surrogate chosen, False otherwise
        """
        return isinstance(my_type, type(ContinuousOutput))

    @model_validator(mode="after")
    def validate_mapping(self):
        """Validate the mapping keys match the categories of the input feature.

        Raises:
            ValueError: If more than one input feature is present.
            ValueError: If the mapping keys do not match the categories of the input feature.
        """
        if len(self.inputs) != 1:
            raise ValueError(
                "Only one input is supported for the `CategoricalDeterministicSurrogate`"
            )
        if sorted(self.inputs[0].categories) != sorted(self.mapping.keys()):
            raise ValueError("Mapping keys do not match input feature keys.")
        return self


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
                "Only numerical inputs are supported for the `LinearDeterministicSurrogate`",
            )
        return self

    @model_validator(mode="after")
    def validate_coefficients(self):
        if sorted(self.inputs.get_keys()) != sorted(self.coefficients.keys()):
            raise ValueError("coefficient keys do not match input feature keys.")
        return self
