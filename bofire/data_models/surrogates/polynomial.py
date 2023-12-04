from typing import Literal

from pydantic import Field

from bofire.data_models.domain.api import Inputs, Outputs
from bofire.data_models.kernels.api import (
    PolynomialKernel,
)
from bofire.data_models.priors.api import BOTORCH_NOISE_PRIOR, AnyPrior

# from bofire.data_models.strategies.api import FactorialStrategy
from bofire.data_models.surrogates.trainable_botorch import TrainableBotorchSurrogate


class PolynomialSurrogate(TrainableBotorchSurrogate):
    type: Literal["PolynomialSurrogate"] = "PolynomialSurrogate"

    kernel: PolynomialKernel = Field(default_factory=lambda: PolynomialKernel(power=2))
    noise_prior: AnyPrior = Field(default_factory=lambda: BOTORCH_NOISE_PRIOR())

    @staticmethod
    def from_power(power: int, inputs: Inputs, outputs: Outputs):
        return PolynomialSurrogate(
            kernel=PolynomialKernel(power=power), inputs=inputs, outputs=outputs
        )
