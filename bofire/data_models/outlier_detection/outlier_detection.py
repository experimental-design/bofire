from typing import Literal

from pydantic import Field
from typing_extensions import Annotated

from bofire.data_models.base import BaseModel
from bofire.data_models.surrogates.api import SingleTaskGPSurrogate


class OutlierDetection(BaseModel):
    type: str


class IterativeTrimming(OutlierDetection):
    type: Literal["IterativeTrimming"] = "IterativeTrimming"
    alpha1: Annotated[float, Field(gt=0.0, lt=1.0)] = 0.5
    alpha2: Annotated[float, Field(gt=0.0, lt=1.0)] = 0.975
    nsh: Annotated[int, Field(ge=1)] = 2
    ncc: Annotated[int, Field(ge=1)] = 2
    nrw: Annotated[int, Field(ge=1)] = 1
    base_gp: SingleTaskGPSurrogate
