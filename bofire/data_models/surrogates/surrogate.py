from abc import abstractmethod
from typing import Optional, Type

from pydantic import Field, field_validator

from bofire.data_models.base import BaseModel
from bofire.data_models.domain.api import Inputs, Outputs
from bofire.data_models.features.api import AnyOutput
from bofire.data_models.types import InputTransformSpecs


class Surrogate(BaseModel):
    type: str
    inputs: Inputs
    outputs: Outputs
    input_preprocessing_specs: InputTransformSpecs = Field(
        default_factory=dict,
        validate_default=True,
    )
    dump: Optional[str] = None

    @field_validator("input_preprocessing_specs")
    @classmethod
    def validate_input_preprocessing_specs(cls, v, info):
        # we also validate the number of input features here
        if len(info.data["inputs"]) == 0:
            raise ValueError("At least one input feature has to be provided.")
        v = info.data["inputs"]._validate_transform_specs(v)
        return v

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
