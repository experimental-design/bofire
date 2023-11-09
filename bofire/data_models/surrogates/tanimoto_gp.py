from typing import Literal

from pydantic import Field, validator

from bofire.data_models.kernels.api import AnyKernel, ScaleKernel
from bofire.data_models.kernels.molecular import TanimotoKernel
from bofire.data_models.priors.api import (
    BOTORCH_NOISE_PRIOR,
    BOTORCH_SCALE_PRIOR,
    AnyPrior,
)
from bofire.data_models.surrogates.botorch import BotorchSurrogate
from bofire.data_models.surrogates.scaler import ScalerEnum
from bofire.data_models.surrogates.trainable import TrainableSurrogate


class TanimotoGPSurrogate(BotorchSurrogate, TrainableSurrogate):
    type: Literal["TanimotoGPSurrogate"] = "TanimotoGPSurrogate"

    kernel: AnyKernel = Field(
        default_factory=lambda: ScaleKernel(
            base_kernel=TanimotoKernel(
                ard=True,
            ),
            outputscale_prior=BOTORCH_SCALE_PRIOR(),
        )
    )
    noise_prior: AnyPrior = Field(default_factory=lambda: BOTORCH_NOISE_PRIOR())
    scaler: ScalerEnum = ScalerEnum.IDENTITY
    output_scaler: ScalerEnum = ScalerEnum.IDENTITY

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
