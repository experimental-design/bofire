from typing import Literal, Type

from pydantic import Field

# from bofire.data_models.strategies.api import FactorialStrategy
from bofire.data_models.features.api import AnyOutput, ContinuousOutput
from bofire.data_models.kernels.api import LinearKernel
from bofire.data_models.priors.api import THREESIX_NOISE_PRIOR, AnyPrior
from bofire.data_models.surrogates.scaler import ScalerEnum
from bofire.data_models.surrogates.trainable_botorch import TrainableBotorchSurrogate


class LinearSurrogate(TrainableBotorchSurrogate):
    type: Literal["LinearSurrogate"] = "LinearSurrogate"

    kernel: LinearKernel = Field(default_factory=lambda: LinearKernel())
    noise_prior: AnyPrior = Field(default_factory=lambda: THREESIX_NOISE_PRIOR())
    scaler: ScalerEnum = ScalerEnum.NORMALIZE

    @classmethod
    def is_output_implemented(cls, my_type: Type[AnyOutput]) -> bool:
        """Abstract method to check output type for surrogate models
        Args:
            my_type: continuous or categorical output
        Returns:
            bool: True if the output type is valid for the surrogate chosen, False otherwise
        """
        return isinstance(my_type, type(ContinuousOutput))
