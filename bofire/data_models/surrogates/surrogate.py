from typing import Optional, Union

from pydantic import Field, validator

from bofire.data_models.base import BaseModel
from bofire.data_models.domain.api import Inputs, Outputs
from bofire.data_models.features.api import ContinuousOutput, CategoricalOutput, TInputTransformSpecs


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
    
    @classmethod # TODO: Remove this, change it, ???
    def is_output_implemented(cls, outputs, my_type: Union[ContinuousOutput, CategoricalOutput]) -> bool:
        """Abstract method to check output type for surrogate models

        Args:
            outputs: objective functions for the surrogate
            my_type: continuous or categorical output

        Returns:
            bool: True if the output type is valid for the surrogate chosen, False otherwise
        """
        for o in outputs:
            if not isinstance(o, my_type):
                return False
        return True
