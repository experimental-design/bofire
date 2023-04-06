from typing import Literal

from pydantic import conint

from bofire.data_models.surrogates.botorch import BotorchSurrogate
from bofire.data_models.surrogates.scaler import ScalerEnum


class SaasSingleTaskGPSurrogate(BotorchSurrogate):
    type: Literal["SaasSingleTaskGPSurrogate"] = "SaasSingleTaskGPSurrogate"
    warmup_steps: conint(ge=1) = 256  # type: ignore
    num_samples: conint(ge=1) = 128  # type: ignore
    thinning: conint(ge=0) = 16  # type: ignore
    scaler: ScalerEnum = ScalerEnum.NORMALIZE
