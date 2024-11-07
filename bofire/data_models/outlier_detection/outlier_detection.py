from abc import abstractmethod
from typing import Annotated, Literal, Union

from pydantic import Field, field_validator

from bofire.data_models.base import BaseModel
from bofire.data_models.domain.api import Inputs, Outputs
from bofire.data_models.surrogates.api import (
    MixedSingleTaskGPSurrogate,
    SingleTaskGPSurrogate,
)


class OutlierDetection(BaseModel):
    type: str

    @property
    @abstractmethod
    def inputs(self) -> Inputs:
        pass

    @property
    @abstractmethod
    def outputs(self) -> Outputs:
        pass


class IterativeTrimming(OutlierDetection):
    """Remove outliers using Robust GP

    Paper: Robust Gaussian Process Regression Based on Iterative Trimming.
    https://arxiv.org/pdf/2011.11057.pdf

    Attributes:
        alpha1 (float in (0, 1)): Trimming parameter.
        alpha2 (float in (0, 1)): Reweighting parameter.
        nsh (int (>=1)): Number of shrinking iterations.
        ncc (int (>=1)): Number of concentrating iterations.
        nrw (int (>=1)): Number of reweighting iterations.
        base_gp (SingleTaskGPSurrogate): Gaussian process model for outlier detection.

    """

    type: Literal["IterativeTrimming"] = "IterativeTrimming"
    alpha1: Annotated[float, Field(gt=0.0, lt=1.0)] = 0.5
    alpha2: Annotated[float, Field(gt=0.0, lt=1.0)] = 0.975
    nsh: Annotated[int, Field(ge=1)] = 2
    ncc: Annotated[int, Field(ge=1)] = 2
    nrw: Annotated[int, Field(ge=1)] = 1
    base_gp: Union[SingleTaskGPSurrogate, MixedSingleTaskGPSurrogate]

    @field_validator("base_gp")
    @classmethod
    def validate_base_gp(cls, v):
        # validate that all base_gps are single output surrogates
        # TODO: this restriction has to be removed at some point

        if len(v.outputs) != 1:
            raise ValueError("Only single output base_gps allowed.")

        return v

    @property
    def inputs(self) -> Inputs:
        return self.base_gp.inputs

    @property
    def outputs(self) -> Outputs:
        return self.base_gp.outputs
