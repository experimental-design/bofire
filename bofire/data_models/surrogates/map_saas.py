from typing import Literal, Type

from pydantic import PositiveInt

from bofire.data_models.features.api import AnyOutput, ContinuousOutput
from bofire.data_models.surrogates.trainable_botorch import TrainableBotorchSurrogate


class TestSurrogate:
    pass


class AdditiveMapSaasSingleTaskGPSurrogate(TrainableBotorchSurrogate):
    """Additive MAP SAAS single-task GP

    Maximum-a-posteriori (MAP) version of the sparse axis-aligned subspace
    `FullyBayesianSingleTaskGPSurrogate` with `model_type` equals to "saas".

    Attributes:
        n_taus (PositiveInt): Number of sub-kernels to use in the SAAS model.
    """

    type: Literal["AdditiveMapSaasSingleTaskGPSurrogate"] = (
        "AdditiveMapSaasSingleTaskGPSurrogate"
    )
    n_taus: PositiveInt = 4

    @classmethod
    def is_output_implemented(cls, my_type: Type[AnyOutput]) -> bool:
        """Abstract method to check output type for surrogate models
        Args:
            my_type: continuous or categorical output
        Returns:
            bool: True if the output type is valid for the surrogate chosen, False otherwise
        """
        return isinstance(my_type, type(ContinuousOutput))
