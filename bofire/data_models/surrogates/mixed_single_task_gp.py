from typing import Literal

from pydantic import Field

from bofire.data_models.domain.api import Constraints
from bofire.data_models.kernels.api import (
    AnyCategoricalKernal,
    AnyContinuousKernel,
    HammondDistanceKernel,
    MaternKernel,
)
from bofire.data_models.surrogates.botorch import BotorchSurrogate
from bofire.data_models.surrogates.single_task_gp import ScalerEnum


class MixedSingleTaskGPSurrogate(BotorchSurrogate):
    type: Literal["MixedSingleTaskGPSurrogate"] = "MixedSingleTaskGPSurrogate"
    constraints: Constraints
    continuous_kernel: AnyContinuousKernel = Field(
        default_factory=lambda: MaternKernel(ard=True, nu=2.5)
    )
    categorical_kernel: AnyCategoricalKernal = Field(
        default_factory=lambda: HammondDistanceKernel(ard=True)
    )
    scaler: ScalerEnum = ScalerEnum.NORMALIZE
