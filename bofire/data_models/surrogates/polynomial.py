from typing import Literal

from pydantic import Field, validator

from bofire.data_models.domain.api import Inputs, Outputs
from bofire.data_models.kernels.api import (
    PolynomialKernel,
)
from bofire.data_models.priors.api import BOTORCH_NOISE_PRIOR, AnyPrior

# from bofire.data_models.strategies.api import FactorialStrategy
from bofire.data_models.surrogates.botorch import BotorchSurrogate
from bofire.data_models.surrogates.scaler import ScalerEnum
from bofire.data_models.surrogates.trainable import TrainableSurrogate


class PolynomialSurrogate(BotorchSurrogate, TrainableSurrogate):
    type: Literal["PolynomialSurrogate"] = "PolynomialSurrogate"

    kernel: PolynomialKernel = Field(default_factory=lambda: PolynomialKernel(power=2))
    noise_prior: AnyPrior = Field(default_factory=lambda: BOTORCH_NOISE_PRIOR())
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

    @staticmethod
    def from_power(power: int, inputs: Inputs, outputs: Outputs):
        return PolynomialSurrogate(
            kernel=PolynomialKernel(power=power), inputs=inputs, outputs=outputs
        )
