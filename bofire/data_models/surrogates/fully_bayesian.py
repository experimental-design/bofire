from typing import Annotated, List, Literal, Type

from pydantic import AfterValidator, Field, field_validator, model_validator

from bofire.data_models.features.api import AnyOutput, ContinuousOutput
from bofire.data_models.surrogates.trainable_botorch import TrainableBotorchSurrogate
from bofire.data_models.types import make_unique_validator


class FullyBayesianSingleTaskGPSurrogate(TrainableBotorchSurrogate):
    type: Literal["FullyBayesianSingleTaskGPSurrogate"] = (
        "FullyBayesianSingleTaskGPSurrogate"
    )
    model_type: Literal["linear", "saas", "hvarfner"] = "saas"
    warmup_steps: Annotated[int, Field(ge=1)] = 256
    num_samples: Annotated[int, Field(ge=1)] = 128
    thinning: Annotated[int, Field(ge=1)] = 16
    features_to_warp: Annotated[
        List[str], AfterValidator(make_unique_validator("Features"))
    ] = []

    @model_validator(mode="after")
    def validate_features_to_warp(self):
        input_keys = self.inputs.get_keys()
        for feature in self.features_to_warp:
            if feature not in input_keys:
                raise ValueError(
                    f"Feature '{feature}' in features_to_warp is not a valid input key."
                )
        return self

    @field_validator("thinning")
    @classmethod
    def validate_thinning(cls, thinning, info):
        if info.data["num_samples"] / thinning < 1:
            raise ValueError("`num_samples` has to be larger than `thinning`.")
        return thinning

    @classmethod
    def is_output_implemented(cls, my_type: Type[AnyOutput]) -> bool:
        """Abstract method to check output type for surrogate models
        Args:
            my_type: continuous or categorical output
        Returns:
            bool: True if the output type is valid for the surrogate chosen, False otherwise
        """
        return isinstance(my_type, type(ContinuousOutput))
