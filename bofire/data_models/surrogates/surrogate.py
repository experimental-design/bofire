from abc import abstractmethod
from typing import Any, Optional, Type

from pydantic import field_validator

from bofire.data_models.base import BaseModel
from bofire.data_models.domain.api import Inputs, Outputs
from bofire.data_models.encodings.api import OrdinalEncoding
from bofire.data_models.features.api import AnyOutput, CategoricalInput
from bofire.data_models.types import InputTransformSpecs


class Surrogate(BaseModel):
    type: Any
    inputs: Inputs
    outputs: Outputs
    dump: Optional[str] = None

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
