from bofire.data_models.acquisition_functions.acquisition_function import (
    AcquisitionFunction,
    MultiObjectiveAcquisitionFunction,
    PreferenceAcquisitionFunction,
    SingleObjectiveAcquisitionFunction,
    pTS,
    qEHVI,
    qEI,
    qEUBO,
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
    qEUBO,
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
    qNEI, qEI, qSR, qUCB, qPI, qLogEI, qLogNEI, qLogPF, pTS
)

AnyMultiObjectiveAcquisitionFunction = tagged_union(
    qEHVI, qLogEHVI, qNEHVI, qLogNEHVI, qMFHVKG
)

AnyActiveLearningAcquisitionFunction = qNegIntPosVar

AnyPreferenceAcquisitionFunction = qEUBO

# Acquisition function that cannot handle constraints intrinsically but fall back to constructing
# a constrainted MC acquisition objective.
AnyUnconstrainedAcquisitionFunction = tagged_union(qSR, qUCB, pTS)
