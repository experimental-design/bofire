from typing import Optional

from pydantic import Field, validator

from bofire.data_models.base import BaseModel
from bofire.data_models.domain.api import Inputs, Outputs
from bofire.data_models.features.api import TInputTransformSpecs


class Surrogate(BaseModel):
    type: str

    inputs: Inputs
    outputs: Outputs
    input_preprocessing_specs: TInputTransformSpecs = Field(default_factory=dict)
    dump: Optional[str] = None

    @validator("input_preprocessing_specs", always=True)
    def validate_input_preprocessing_specs(cls, v, values):
        # we also validate the number of input features here
        if len(values["inputs"]) == 0:
            raise ValueError("At least one input feature has to be provided.")
        v = values["inputs"]._validate_transform_specs(v)
        return v

    @validator("outputs")
    def validate_outputs(cls, v, values):
        if len(v) == 0:
            raise ValueError("At least one output feature has to be provided.")
        return v
