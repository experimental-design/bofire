from typing import Union

from bofire.data_models.acquisition_functions.acquisition_function import (
    AcquisitionFunction,
    MultiObjectiveAcquisitionFunction,
    SingleObjectiveAcquisitionFunction,
    pTS,
    qEHVI,
    qEI,
    qLogEHVI,
    qLogEI,
    qLogNEHVI,
    qLogNEI,
    qLogPF,
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
    qLogPF,
    pTS,
]

AnySingleObjectiveAcquisitionFunction = Union[
    qNEI, qEI, qSR, qUCB, qPI, qLogEI, qLogNEI, qLogPF, pTS
]

AnyMultiObjectiveAcquisitionFunction = Union[qEHVI, qLogEHVI, qNEHVI, qLogNEHVI]

AnyActiveLearningAcquisitionFunction = qNegIntPosVar

# Acquisition function that cannot handle constraints intrinsically but fall back to constructing
# a constrainted MC acquisition objective.
AnyUnconstrainedAcquisitionFunction = Union[qSR, qUCB, pTS]
