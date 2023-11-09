from typing import Literal

from pydantic import conint, validator

from bofire.data_models.surrogates.botorch import BotorchSurrogate
from bofire.data_models.surrogates.scaler import ScalerEnum
from bofire.data_models.surrogates.trainable import TrainableSurrogate


class SaasSingleTaskGPSurrogate(BotorchSurrogate, TrainableSurrogate):
    type: Literal["SaasSingleTaskGPSurrogate"] = "SaasSingleTaskGPSurrogate"
    warmup_steps: conint(ge=1) = 256  # type: ignore
    num_samples: conint(ge=1) = 128  # type: ignore
    thinning: conint(ge=1) = 16  # type: ignore
    scaler: ScalerEnum = ScalerEnum.NORMALIZE
    output_scaler: ScalerEnum = ScalerEnum.STANDARDIZE

    @validator("output_scaler")
    def validate_output_scaler(cls, output_scaler):
        """validates that output_scaler is a valid type

        Args:
            output_scaler (ScalerEnum): Scaler used to transform the output

        Raises:
            ValueError: when ScalerEnum.NORMALIZE is used

        Returns:
            ScalerEnum: Scaler used to transform the output
        """
        if output_scaler == ScalerEnum.NORMALIZE:
            raise ValueError("Normalize is not supported as an output transform.")

        return output_scaler

    @validator("thinning")
    def validate_thinning(cls, value, values):
        if values["num_samples"] / value < 1:
            raise ValueError("`num_samples` has to be larger than `thinning`.")
        return value
