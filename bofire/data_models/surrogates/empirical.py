from typing import Literal

from bofire.data_models.surrogates.botorch import BotorchSurrogate


class EmpiricalSurrogate(BotorchSurrogate):
    type: Literal["EmpiricalSurrogate"] = "EmpiricalSurrogate"
