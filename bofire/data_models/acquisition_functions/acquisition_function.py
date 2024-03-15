from typing import Annotated, Dict, Literal, Optional

from pydantic import Field, PositiveFloat

from bofire.data_models.base import BaseModel


class AcquisitionFunction(BaseModel):
    type: str


class SingleObjectiveAcquisitionFunction(AcquisitionFunction):
    type: str


class MultiObjectiveAcquisitionFunction(AcquisitionFunction):
    type: str


class qNEI(SingleObjectiveAcquisitionFunction):
    type: Literal["qNEI"] = "qNEI"
    prune_baseline: bool = True


class qLogNEI(SingleObjectiveAcquisitionFunction):
    type: Literal["qLogNEI"] = "qLogNEI"
    prune_baseline: bool = True


class qEI(SingleObjectiveAcquisitionFunction):
    type: Literal["qEI"] = "qEI"


class qLogEI(SingleObjectiveAcquisitionFunction):
    type: Literal["qLogEI"] = "qLogEI"


class qSR(SingleObjectiveAcquisitionFunction):
    type: Literal["qSR"] = "qSR"


class qUCB(SingleObjectiveAcquisitionFunction):
    type: Literal["qUCB"] = "qUCB"
    beta: Annotated[float, Field(ge=0)] = 0.2


class qPI(SingleObjectiveAcquisitionFunction):
    type: Literal["qPI"] = "qPI"
    tau: PositiveFloat = 1e-3


class qEHVI(MultiObjectiveAcquisitionFunction):
    type: Literal["qEHVI"] = "qEHVI"
    alpha: Annotated[float, Field(ge=0)] = 0.0


class qLogEHVI(MultiObjectiveAcquisitionFunction):
    type: Literal["qLogEHVI"] = "qLogEHVI"
    alpha: Annotated[float, Field(ge=0)] = 0.0


class qNEHVI(MultiObjectiveAcquisitionFunction):
    type: Literal["qNEHVI"] = "qNEHVI"
    alpha: Annotated[float, Field(ge=0)] = 0.0
    prune_baseline: bool = True


class qLogNEHVI(MultiObjectiveAcquisitionFunction):
    type: Literal["qLogNEHVI"] = "qLogNEHVI"
    alpha: Annotated[float, Field(ge=0)] = 0.0
    prune_baseline: bool = True


class qNegIntPosVar(SingleObjectiveAcquisitionFunction):
    type: Literal["qNegIntPosVar"] = "qNegIntPosVar"
    n_mc_points: Annotated[int, Field(ge=1)] = 128
    weights: Optional[Dict[str, PositiveFloat]] = Field(default_factory=lambda: None)
