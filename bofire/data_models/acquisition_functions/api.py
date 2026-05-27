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
    qLogPF,
    qMFHVKG,
    qNegIntPosVar,
    qNEHVI,
    qNEI,
    qPI,
    qSR,
    qUCB,
)
from bofire.data_models.unions import tagged_union


AnyAcquisitionFunction = tagged_union(
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
    qMFHVKG,
    qNegIntPosVar,
    qLogPF,
)

AnySingleObjectiveAcquisitionFunction = tagged_union(
    qNEI, qEI, qSR, qUCB, qPI, qLogEI, qLogNEI, qLogPF
)

AnyMultiObjectiveAcquisitionFunction = tagged_union(
    qEHVI, qLogEHVI, qNEHVI, qLogNEHVI, qMFHVKG
)

AnyActiveLearningAcquisitionFunction = qNegIntPosVar
