from typing import Literal

from pydantic import Field

from bofire.data_models.kernels.api import LinearKernel
from bofire.data_models.priors.api import BOTORCH_NOISE_PRIOR, AnyPrior

# from bofire.data_models.strategies.api import FactorialStrategy
from bofire.data_models.surrogates.trainable_botorch import TrainableBotorchSurrogate


class LinearSurrogate(TrainableBotorchSurrogate):
    type: Literal["LinearSurrogate"] = "LinearSurrogate"

    kernel: LinearKernel = Field(default_factory=lambda: LinearKernel())
    noise_prior: AnyPrior = Field(default_factory=lambda: BOTORCH_NOISE_PRIOR())
