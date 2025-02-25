from typing import Union

from bofire.data_models.acquisition_functions.acquisition_function import (
    AcquisitionFunction,
    MultiFideltyAcquisitionFunction,
    MultiObjectiveAcquisitionFunction,
    SingleObjectiveAcquisitionFunction,
    qEHVI,
    qEI,
    qLogEHVI,
    qLogEI,
    qLogNEHVI,
    qLogNEI,
    qMFGibbon,
    qMFMES,
    qMFVariance,
    qNegIntPosVar,
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
    MultiFideltyAcquisitionFunction,
]

AnyAcquisitionFunction = Union[
    qNEI,
    qEI,
    qSR,
    qUCB,
    qPI,
    qLogEI,
    qLogNEI,
    qEHVI,
    qLogEHVI,
    qNEHVI,
    qLogNEHVI,
    qNegIntPosVar,
]

AnySingleObjectiveAcquisitionFunction = Union[
    qNEI,
    qEI,
    qSR,
    qUCB,
    qPI,
    qLogEI,
    qLogNEI,
]

AnyMultiObjectiveAcquisitionFunction = Union[qEHVI, qLogEHVI, qNEHVI, qLogNEHVI]

AnyActiveLearningAcquisitionFunction = qNegIntPosVar

AnyMultiFidelityAcquisitionFunction = Union[qMFMES, qMFGibbon, qMFVariance]
