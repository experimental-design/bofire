from typing import Literal

from pydantic import Field

from bofire.data_models.kernels.api import AnyKernel, MaternKernel, ScaleKernel
from bofire.data_models.priors.api import BOTORCH_LENGTHCALE_PRIOR, BOTORCH_SCALE_PRIOR
from bofire.data_models.surrogates.botorch import BotorchSurrogate
from bofire.data_models.surrogates.scaler import ScalerEnum


class SingleTaskGPSurrogate(BotorchSurrogate):
    type: Literal["SingleTaskGPSurrogate"] = "SingleTaskGPSurrogate"

    kernel: AnyKernel = Field(
        default_factory=lambda: ScaleKernel(
            base_kernel=MaternKernel(
                ard=True,
                nu=2.5,
                lengthscale_prior=BOTORCH_LENGTHCALE_PRIOR,
            ),
            outputscale_prior=BOTORCH_SCALE_PRIOR,
        )
    )
    scaler: ScalerEnum = ScalerEnum.NORMALIZE
