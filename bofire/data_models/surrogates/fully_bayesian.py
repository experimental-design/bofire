from typing import Literal

from pydantic import conint, validator

from bofire.data_models.surrogates.botorch import BotorchSurrogate
from bofire.data_models.surrogates.scaler import ScalerEnum


class SaasSingleTaskGPSurrogate(BotorchSurrogate):
    type: Literal["SaasSingleTaskGPSurrogate"] = "SaasSingleTaskGPSurrogate"
    warmup_steps: conint(ge=1) = 256  # type: ignore
    num_samples: conint(ge=1) = 128  # type: ignore
    thinning: conint(ge=1) = 16  # type: ignore
    scaler: ScalerEnum = ScalerEnum.NORMALIZE

    @validator("thinning")
    def validate_thinning(cls, value, values):
        if values["num_samples"] / value < 1:
            raise ValueError("`num_samples` has to be larger than `thinning`.")
        return value
