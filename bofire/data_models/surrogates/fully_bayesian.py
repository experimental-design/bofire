from typing import Literal, Type

from pydantic import conint, validator

from bofire.data_models.features.api import AnyOutput, ContinuousOutput
from bofire.data_models.surrogates.trainable_botorch import TrainableBotorchSurrogate


class SaasSingleTaskGPSurrogate(TrainableBotorchSurrogate):
    type: Literal["SaasSingleTaskGPSurrogate"] = "SaasSingleTaskGPSurrogate"
    warmup_steps: conint(ge=1) = 256  # type: ignore
    num_samples: conint(ge=1) = 128  # type: ignore
    thinning: conint(ge=1) = 16  # type: ignore

    @validator("thinning")
    def validate_thinning(cls, value, values):
        if values["num_samples"] / value < 1:
            raise ValueError("`num_samples` has to be larger than `thinning`.")
        return value

    @classmethod
    def is_output_implemented(cls, my_type: Type[AnyOutput]) -> bool:
        """Abstract method to check output type for surrogate models

        Args:
            outputs: objective functions for the surrogate
            my_type: continuous or categorical output

        Returns:
            bool: True if the output type is valid for the surrogate chosen, False otherwise
        """
        return True if isinstance(my_type, ContinuousOutput) else False
