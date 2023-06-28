from typing import Literal

from pydantic import Field
from typing_extensions import Annotated

from bofire.data_models.base import BaseModel
from bofire.data_models.surrogates.api import SingleTaskGPSurrogate


class OutlierDetection(BaseModel):
    type: str


class IterativeTrimming(OutlierDetection):
    """
    Paper: Robust Gaussian Process Regression Based on Iterative Trimming.
    https://arxiv.org/pdf/2011.11057.pdf

    Parameters
    ----------

    alpha1, alpha2: float in (0, 1)
        Trimming and reweighting parameters respectively.
    nsh, ncc, nrw: int (>=1)
        Number of shrinking, concentrating, and reweighting iterations respectively.
    bas_gp: SingleTaskGPSurrogate = Gaussian process model for outlier detection.
    """

    type: Literal["IterativeTrimming"] = "IterativeTrimming"
    alpha1: Annotated[float, Field(gt=0.0, lt=1.0)] = 0.5
    alpha2: Annotated[float, Field(gt=0.0, lt=1.0)] = 0.975
    nsh: Annotated[int, Field(ge=1)] = 2
    ncc: Annotated[int, Field(ge=1)] = 2
    nrw: Annotated[int, Field(ge=1)] = 1
    base_gp: SingleTaskGPSurrogate
