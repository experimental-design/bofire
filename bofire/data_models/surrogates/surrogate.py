from abc import abstractmethod
from typing import Any, Optional, Type

from pydantic import Field, field_validator

from bofire.data_models.base import BaseModel
from bofire.data_models.domain.api import Inputs, Outputs
from bofire.data_models.encodings.api import OrdinalEncoding
from bofire.data_models.features.api import AnyOutput, CategoricalInput
from bofire.data_models.types import InputTransformSpecs


class Surrogate(BaseModel):
    """Model of the relation between the inputs and the outputs."""

    type: Any
    inputs: Inputs = Field(
        description="Input features the surrogate acts on. When the surrogate is used "
        "by a strategy, these may be a subset of the strategy's inputs, so that "
        "different outputs can be modelled from different inputs.",
    )
    outputs: Outputs = Field(
        description="Output features the surrogate predicts. Most surrogates take "
        "exactly one.",
    )
    dump: Optional[str] = Field(
        default=None,
        description="The fitted model, serialized, so a trained surrogate survives a "
        "round trip. Written when the surrogate is dumped; not set by hand.",
    )

    @property
    def input_preprocessing_specs(self) -> InputTransformSpecs:
        """Pre-model tensorization: every categorical is ordinal-encoded (the in-model
        encoding choice lives in ``categorical_encodings``)."""
        return {
            key: OrdinalEncoding()
            for key in self.inputs.get_keys(CategoricalInput, exact=False)
        }

    @field_validator("inputs")
    @classmethod
    def validate_inputs_not_empty(cls, inputs):
        if len(inputs) == 0:
            raise ValueError("At least one input feature has to be provided.")
        return inputs

    @field_validator("outputs")
    @classmethod
    def validate_outputs(cls, outputs, info):
        if len(outputs) == 0:
            raise ValueError("At least one output feature has to be provided.")
        for o in outputs:
            if not cls.is_output_implemented(type(o)):
                raise ValueError("Invalid output type passed.")
        return outputs

    @classmethod
    @abstractmethod
    def is_output_implemented(cls, my_type: Type[AnyOutput]) -> bool:
        """Abstract method to check output type for surrogate models
        Args:
            outputs: objective functions for the surrogate
            my_type: continuous or categorical output
        Returns:
            bool: True if the output type is valid for the surrogate chosen, False otherwise
        """
