from typing import Literal

from pydantic import Field, validator

from bofire.data_models.enum import CategoricalEncodingEnum
from bofire.data_models.kernels.api import (
    AnyCategoricalKernal,
    AnyContinuousKernel,
    HammondDistanceKernel,
    MaternKernel,
)
from bofire.data_models.surrogates.botorch import BotorchSurrogate
from bofire.data_models.surrogates.single_task_gp import ScalerEnum
from bofire.data_models.surrogates.trainable import TrainableSurrogate


class MixedSingleTaskGPSurrogate(BotorchSurrogate, TrainableSurrogate):
    type: Literal["MixedSingleTaskGPSurrogate"] = "MixedSingleTaskGPSurrogate"
    continuous_kernel: AnyContinuousKernel = Field(
        default_factory=lambda: MaternKernel(ard=True, nu=2.5)
    )
    categorical_kernel: AnyCategoricalKernal = Field(
        default_factory=lambda: HammondDistanceKernel(ard=True)
    )
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

    @validator("input_preprocessing_specs")
    def validate_categoricals(cls, v, values):
        """Checks that at least one one-hot encoded categorical feauture is present."""
        if CategoricalEncodingEnum.ONE_HOT not in v.values():
            raise ValueError(
                "MixedSingleTaskGPSurrogate can only be used if at least one one-hot encoded categorical feature is present."
            )
        return v
