from typing import Optional

from pydantic import Field, field_validator

from bofire.data_models.base import BaseModel
from bofire.data_models.domain.api import Inputs, Outputs
from bofire.data_models.types import TInputTransformSpecs


class Surrogate(BaseModel):
    type: str

    inputs: Inputs
    outputs: Outputs
    input_preprocessing_specs: TInputTransformSpecs = Field(
        default_factory=dict, validate_default=True
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
    def validate_outputs(cls, v, values):
        if len(v) == 0:
            raise ValueError("At least one output feature has to be provided.")
        return v
