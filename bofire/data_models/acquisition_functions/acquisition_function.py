from typing import Annotated, Dict, Literal, Optional

from pydantic import Field, PositiveFloat

from bofire.data_models.base import BaseModel
from bofire.data_models.types import IntPowerOfTwo


class AcquisitionFunction(BaseModel):
    type: str


class SingleObjectiveAcquisitionFunction(AcquisitionFunction):
    type: str


class MultiObjectiveAcquisitionFunction(AcquisitionFunction):
    type: str


class qNEI(SingleObjectiveAcquisitionFunction):
    type: Literal["qNEI"] = "qNEI"
    prune_baseline: bool = True
    n_mc_samples: IntPowerOfTwo = 512


class qLogNEI(SingleObjectiveAcquisitionFunction):
    type: Literal["qLogNEI"] = "qLogNEI"
    prune_baseline: bool = True
    n_mc_samples: IntPowerOfTwo = 512


class qEI(SingleObjectiveAcquisitionFunction):
    type: Literal["qEI"] = "qEI"
    n_mc_samples: IntPowerOfTwo = 512


class qLogEI(SingleObjectiveAcquisitionFunction):
    type: Literal["qLogEI"] = "qLogEI"
    n_mc_samples: IntPowerOfTwo = 512


class qSR(SingleObjectiveAcquisitionFunction):
    type: Literal["qSR"] = "qSR"
    n_mc_samples: IntPowerOfTwo = 512


class qUCB(SingleObjectiveAcquisitionFunction):
    type: Literal["qUCB"] = "qUCB"
    beta: Annotated[float, Field(ge=0)] = 0.2
    n_mc_samples: IntPowerOfTwo = 512


class qPI(SingleObjectiveAcquisitionFunction):
    type: Literal["qPI"] = "qPI"
    tau: PositiveFloat = 1e-3
    n_mc_samples: IntPowerOfTwo = 512


class qEHVI(MultiObjectiveAcquisitionFunction):
    type: Literal["qEHVI"] = "qEHVI"
    alpha: Annotated[float, Field(ge=0)] = 0.0
    n_mc_samples: IntPowerOfTwo = 512


class qLogEHVI(MultiObjectiveAcquisitionFunction):
    type: Literal["qLogEHVI"] = "qLogEHVI"
    alpha: Annotated[float, Field(ge=0)] = 0.0
    n_mc_samples: IntPowerOfTwo = 512


class qNEHVI(MultiObjectiveAcquisitionFunction):
    type: Literal["qNEHVI"] = "qNEHVI"
    alpha: Annotated[float, Field(ge=0)] = 0.0
    prune_baseline: bool = True
    n_mc_samples: IntPowerOfTwo = 512


class qLogNEHVI(MultiObjectiveAcquisitionFunction):
    type: Literal["qLogNEHVI"] = "qLogNEHVI"
    alpha: Annotated[float, Field(ge=0)] = 0.0
    prune_baseline: bool = True
    n_mc_samples: IntPowerOfTwo = 512


class qNegIntPosVar(SingleObjectiveAcquisitionFunction):
    type: Literal["qNegIntPosVar"] = "qNegIntPosVar"
    n_mc_samples: IntPowerOfTwo = 512
    weights: Optional[Dict[str, PositiveFloat]] = Field(default_factory=lambda: None)
