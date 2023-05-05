from typing import Literal

from pydantic import Field

from bofire.data_models.kernels.api import AnyKernel, TanimotoKernel, ScaleKernel
from bofire.data_models.priors.api import BOTORCH_LENGTHCALE_PRIOR, BOTORCH_SCALE_PRIOR
from bofire.data_models.surrogates.botorch import BotorchSurrogate
from bofire.data_models.surrogates.scaler import ScalerEnum


class TanimotoGPSurrogate(BotorchSurrogate):
    type: Literal["TanimotoGPSurrogate"] = "TanimotoGPSurrogate"

    kernel: AnyKernel = Field(
        default_factory=lambda: ScaleKernel(
            base_kernel=TanimotoKernel())
    )
