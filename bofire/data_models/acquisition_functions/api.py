from typing import Union

from bofire.data_models.acquisition_functions.acquisition_function import (
    AcquisitionFunction,
    qEI,
    qNEI,
    qPI,
    qSR,
    qUCB,
)

AbstractAcquisitionFunction = AcquisitionFunction

AnyAcquisitionFunction = Union[
    qNEI,
    qEI,
    qSR,
    qUCB,
    qPI,
]
