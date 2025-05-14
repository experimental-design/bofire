from enum import Enum
from typing import Literal, Type

from bofire.data_models.features.api import AnyOutput, ContinuousOutput
from bofire.data_models.surrogates.trainable_botorch import TrainableBotorchSurrogate


class Parametrization(Enum):
    DENSE = "dense"
    DIAGONAL = "diagonal"
    LOWRANK = "lowrank"
    DENSE_PREDICTION = "dense_precision"


class VBLL(TrainableBotorchSurrogate):
    type: Literal["VBLL"] = "VBLL"
    hidden_features: int = 64
    num_layers: int = 3
    parameterization: Parametrization = Parametrization.DENSE
    prior_scale: float = 1.0
    wishart_scale: float = 0.01
    clamp_noise_init: bool = True
    kl_scale: float = 1.0

    @classmethod
    def is_output_implemented(cls, my_type: Type[AnyOutput]) -> bool:
        """Abstract method to check output type for surrogate models
        Args:
            my_type: continuous or categorical output
        Returns:
            bool: True if the output type is valid for the surrogate chosen, False otherwise
        """
        return isinstance(my_type, type(ContinuousOutput))
