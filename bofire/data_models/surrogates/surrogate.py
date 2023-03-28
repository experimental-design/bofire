from pydantic import Field, validator

from bofire.data_models.base import BaseModel
from bofire.data_models.domain.api import Inputs, Outputs
from bofire.data_models.features.api import TInputTransformSpecs


class Surrogate(BaseModel):
    """A class to represent a Surrogate model that is a subclass of BaseModel.

    Attributes:
        type (str): A string representing the type of the model.

        input_features (Inputs): An instance of the Inputs class that represents the input features of the model.
        output_features (Outputs): An instance of the Outputs class that represents the output features of the model.
        input_preprocessing_specs (TInputTransformSpecs): A dictionary that specifies the input preprocessing specifications.
    Methods:
        validate_input_preprocessing_specs(v, values):
            A validator method that validates the input preprocessing specifications.
            It raises a ValueError if there are no input features provided.

        validate_output_features(v, values):
            A validator method that validates the output features.
            It raises a ValueError if there are no output features provided.
    """

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
