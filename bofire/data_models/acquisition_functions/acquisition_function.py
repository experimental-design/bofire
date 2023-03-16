from typing import Literal

from pydantic import PositiveFloat

from bofire.data_models.base import BaseModel


class AcquisitionFunction(BaseModel):
    type: str


class qNEI(AcquisitionFunction):
    type: Literal["qNEI"] = "qNEI"


class qEI(AcquisitionFunction):
    type: Literal["qEI"] = "qEI"


class qSR(AcquisitionFunction):
    type: Literal["qSR"] = "qSR"


class qUCB(AcquisitionFunction):
    type: Literal["qUCB"] = "qUCB"
    beta: PositiveFloat = 0.2


class qPI(AcquisitionFunction):
    type: Literal["qPI"] = "qPI"
    tau: PositiveFloat = 1e-3
