from typing import Annotated, Union

from pydantic import Field

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

AnyAcquisitionFunction = Annotated[
    Union[
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
    ],
    Field(discriminator="type"),
]

AnySingleObjectiveAcquisitionFunction = Annotated[
    Union[qNEI, qEI, qSR, qUCB, qPI, qLogEI, qLogNEI, qLogPF],
    Field(discriminator="type"),
]

AnyMultiObjectiveAcquisitionFunction = Annotated[
    Union[qEHVI, qLogEHVI, qNEHVI, qLogNEHVI],
    Field(discriminator="type"),
]

AnyActiveLearningAcquisitionFunction = qNegIntPosVar
