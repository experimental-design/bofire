from typing import Literal, Type

from pydantic import Field

from bofire.data_models.domain.api import Inputs, Outputs
from bofire.data_models.features.api import AnyOutput, ContinuousOutput
from bofire.data_models.kernels.api import PolynomialKernel
from bofire.data_models.priors.api import THREESIX_NOISE_PRIOR, AnyPrior
from bofire.data_models.surrogates.trainable_botorch import TrainableBotorchSurrogate


class PolynomialSurrogate(TrainableBotorchSurrogate):
    type: Literal["PolynomialSurrogate"] = "PolynomialSurrogate"

    kernel: PolynomialKernel = Field(default_factory=lambda: PolynomialKernel(power=2))
    noise_prior: AnyPrior = Field(default_factory=lambda: THREESIX_NOISE_PRIOR())

    @staticmethod
    def from_power(power: int, inputs: Inputs, outputs: Outputs):
        return PolynomialSurrogate(
            kernel=PolynomialKernel(power=power),
            inputs=inputs,
            outputs=outputs,
        )

    @classmethod
    def is_output_implemented(cls, my_type: Type[AnyOutput]) -> bool:
        """Abstract method to check output type for surrogate models
        Args:
            my_type: continuous or categorical output
        Returns:
            bool: True if the output type is valid for the surrogate chosen, False otherwise
        """
        return isinstance(my_type, type(ContinuousOutput))
