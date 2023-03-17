from pydantic import Field, validator

from bofire.data_models.base import BaseModel
from bofire.data_models.domain.api import Inputs, Outputs
from bofire.data_models.features.api import TInputTransformSpecs


class Surrogate(BaseModel):
    type: str

    input_features: Inputs
    output_features: Outputs
    input_preprocessing_specs: TInputTransformSpecs = Field(default_factory=dict)

    @validator("input_preprocessing_specs", always=True)
    def validate_input_preprocessing_specs(cls, v, values):
        # we also validate the number of input features here
        if len(values["input_features"]) == 0:
            raise ValueError("At least one input feature has to be provided.")
        v = values["input_features"]._validate_transform_specs(v)
        return v

    @validator("output_features")
    def validate_output_features(cls, v, values):
        if len(v) == 0:
            raise ValueError("At least one output feature has to be provided.")
        return v
