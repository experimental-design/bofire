from typing import Literal, Type

from bofire.data_models.features.api import AnyOutput
from bofire.data_models.surrogates.botorch import BotorchSurrogate


class EmpiricalSurrogate(BotorchSurrogate):
    type: Literal["EmpiricalSurrogate"] = "EmpiricalSurrogate"

    @classmethod
    def is_output_implemented(cls, my_type: Type[AnyOutput]) -> bool:
        """Abstract method to check output type for surrogate models
        Args:
            my_type: continuous or categorical output
        Returns:
            bool: True if the output type is valid for the surrogate chosen, False otherwise
        """
        return True
