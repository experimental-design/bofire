from typing import Literal, Optional, Type

from pydantic import Field

from bofire.data_models.features.api import AnyOutput
from bofire.data_models.features.continuous import ContinuousOutput
from bofire.data_models.kernels.api import InfiniteWidthBNNKernel
from bofire.data_models.priors.api import HVARFNER_NOISE_PRIOR, AnyPrior
from bofire.data_models.surrogates.single_task_gp import TrainableBotorchSurrogate
from bofire.data_models.surrogates.trainable import Hyperconfig


class SingleTaskIBNNSurrogate(TrainableBotorchSurrogate):
    type: Literal["SingleTaskIBNNSurrogate"] = "SingleTaskIBNNSurrogate"
    kernel: InfiniteWidthBNNKernel = InfiniteWidthBNNKernel()
    hyperconfig: Optional[Hyperconfig] = None
    noise_prior: AnyPrior = Field(default_factory=lambda: HVARFNER_NOISE_PRIOR())

    @classmethod
    def is_output_implemented(cls, my_type: Type[AnyOutput]) -> bool:
        """Abstract method to check output type for surrogate models
        Args:
            my_type: continuous or categorical output
        Returns:
            bool: True if the output type is valid for the surrogate chosen, False otherwise
        """
        return isinstance(my_type, type(ContinuousOutput))

    @property
    def hyperconfig_access(self) -> Optional[Hyperconfig]:
        return self.hyperconfig
