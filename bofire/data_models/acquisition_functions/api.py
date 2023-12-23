from typing import Union

from bofire.data_models.acquisition_functions.acquisition_function import (
    AcquisitionFunction,
    MultiObjectiveAcquisitionFunction,
    SingleObjectiveAcquisitionFunction,
    qEHVI,
    qEI,
    qLogEHVI,
    qLogEI,
    qLogNEHVI,
    qLogNEI,
    qNEHVI,
    qNEI,
    qPI,
    qSR,
    qUCB,
)

AbstractAcquisitionFunction = [
    AcquisitionFunction,
    SingleObjectiveAcquisitionFunction,
    MultiObjectiveAcquisitionFunction,
]

AnyAcquisitionFunction = Union[
    qNEI, qEI, qSR, qUCB, qPI, qLogEI, qLogNEI, qEHVI, qLogEHVI, qNEHVI, qLogNEHVI
]

AnySingleObjectiveAcquisitionFunction = Union[
    qNEI, qEI, qSR, qUCB, qPI, qLogEI, qLogNEI
]

AnyMultiObjectiveAcquisitionFunction = Union[qEHVI, qLogEHVI, qNEHVI, qLogNEHVI]
