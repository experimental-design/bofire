from typing import Literal

from pydantic import Field

from bofire.data_models.kernels.api import LinearKernel
from bofire.data_models.priors.api import (
    BOTORCH_NOISE_PRIOR,
    BOTORCH_SCALE_PRIOR,
    AnyPrior,
)

# from bofire.data_models.strategies.api import FactorialStrategy
from bofire.data_models.surrogates.botorch import BotorchSurrogate
from bofire.data_models.surrogates.scaler import ScalerEnum
from bofire.data_models.surrogates.trainable import TrainableSurrogate


class LinearSurrogate(BotorchSurrogate, TrainableSurrogate):
    type: Literal["LinearSurrogate"] = "LinearSurrogate"

    kernel: LinearKernel = Field(
        default_factory=lambda: LinearKernel(variance_prior=BOTORCH_SCALE_PRIOR())
    )
    noise_prior: AnyPrior = Field(default_factory=lambda: BOTORCH_NOISE_PRIOR())
    scaler: ScalerEnum = ScalerEnum.NORMALIZE
