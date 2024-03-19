from typing import Annotated, Literal

from pydantic import Field, PositiveFloat

from bofire.data_models.base import BaseModel
from bofire.data_models.types import IntPowerOfTwo


class AcquisitionFunction(BaseModel):
    type: str


class SingleObjectiveAcquisitionFunction(AcquisitionFunction):
    type: str


class MultiObjectiveAcquisitionFunction(AcquisitionFunction):
    type: str


class MCAcquisitionFunction(BaseModel):
    n_mc_samples: IntPowerOfTwo = 512


class qNEI(SingleObjectiveAcquisitionFunction, MCAcquisitionFunction):
    type: Literal["qNEI"] = "qNEI"
    prune_baseline: bool = True


class qLogNEI(SingleObjectiveAcquisitionFunction, MCAcquisitionFunction):
    type: Literal["qLogNEI"] = "qLogNEI"
    prune_baseline: bool = True


class qEI(SingleObjectiveAcquisitionFunction, MCAcquisitionFunction):
    type: Literal["qEI"] = "qEI"


class qLogEI(SingleObjectiveAcquisitionFunction, MCAcquisitionFunction):
    type: Literal["qLogEI"] = "qLogEI"


class qSR(SingleObjectiveAcquisitionFunction, MCAcquisitionFunction):
    type: Literal["qSR"] = "qSR"


class qUCB(SingleObjectiveAcquisitionFunction, MCAcquisitionFunction):
    type: Literal["qUCB"] = "qUCB"
    beta: Annotated[float, Field(ge=0)] = 0.2


class qPI(SingleObjectiveAcquisitionFunction, MCAcquisitionFunction):
    type: Literal["qPI"] = "qPI"
    tau: PositiveFloat = 1e-3


class qEHVI(MultiObjectiveAcquisitionFunction, MCAcquisitionFunction):
    type: Literal["qEHVI"] = "qEHVI"
    alpha: Annotated[float, Field(ge=0)] = 0.0


class qLogEHVI(MultiObjectiveAcquisitionFunction, MCAcquisitionFunction):
    type: Literal["qLogEHVI"] = "qLogEHVI"
    alpha: Annotated[float, Field(ge=0)] = 0.0


class qNEHVI(MultiObjectiveAcquisitionFunction, MCAcquisitionFunction):
    type: Literal["qNEHVI"] = "qNEHVI"
    alpha: Annotated[float, Field(ge=0)] = 0.0
    prune_baseline: bool = True


class qLogNEHVI(MultiObjectiveAcquisitionFunction, MCAcquisitionFunction):
    type: Literal["qLogNEHVI"] = "qLogNEHVI"
    alpha: Annotated[float, Field(ge=0)] = 0.0
    prune_baseline: bool = True
