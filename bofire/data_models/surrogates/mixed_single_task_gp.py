from typing import Literal

from pydantic import Field, validator

from bofire.data_models.enum import CategoricalEncodingEnum
from bofire.data_models.kernels.api import (
    AnyCategoricalKernal,
    AnyContinuousKernel,
    HammondDistanceKernel,
    MaternKernel,
)
from bofire.data_models.surrogates.trainable_botorch import TrainableBotorchSurrogate


class MixedSingleTaskGPSurrogate(TrainableBotorchSurrogate):
    type: Literal["MixedSingleTaskGPSurrogate"] = "MixedSingleTaskGPSurrogate"
    continuous_kernel: AnyContinuousKernel = Field(
        default_factory=lambda: MaternKernel(ard=True, nu=2.5)
    )
    categorical_kernel: AnyCategoricalKernal = Field(
        default_factory=lambda: HammondDistanceKernel(ard=True)
    )

    @validator("input_preprocessing_specs")
    def validate_categoricals(cls, v, values):
        """Checks that at least one one-hot encoded categorical feauture is present."""
        if CategoricalEncodingEnum.ONE_HOT not in v.values():
            raise ValueError(
                "MixedSingleTaskGPSurrogate can only be used if at least one one-hot encoded categorical feature is present."
            )
        return v
