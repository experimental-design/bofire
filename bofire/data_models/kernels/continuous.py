from typing import List, Literal, Optional, Union

from pydantic import Field, PositiveInt, field_validator, model_validator

from bofire.data_models.kernels.kernel import FeatureSpecificKernel
from bofire.data_models.priors.api import (
    HVARFNER_LENGTHSCALE_PRIOR,
    AnyGeneralPrior,
    AnyPrior,
    AnyPriorConstraint,
)


class ContinuousKernel(FeatureSpecificKernel):
    pass


class RBFKernel(ContinuousKernel):
    type: Literal["RBFKernel"] = "RBFKernel"
    ard: bool = True
    lengthscale_prior: Optional[AnyPrior] = None
    lengthscale_constraint: Optional[AnyPriorConstraint] = None


class MaternKernel(ContinuousKernel):
    type: Literal["MaternKernel"] = "MaternKernel"
    ard: bool = True
    nu: float = 2.5
    lengthscale_prior: Optional[AnyPrior] = None
    lengthscale_constraint: Optional[AnyPriorConstraint] = None

    @field_validator("nu")
    def validate_nu(cls, nu):
        if nu not in {0.5, 1.5, 2.5}:
            raise ValueError("nu expected to be 0.5, 1.5, or 2.5")
        return nu


class LinearKernel(ContinuousKernel):
    type: Literal["LinearKernel"] = "LinearKernel"
    variance_prior: Optional[AnyGeneralPrior] = None


class PolynomialKernel(ContinuousKernel):
    type: Literal["PolynomialKernel"] = "PolynomialKernel"
    offset_prior: Optional[AnyGeneralPrior] = None
    power: int = 2


class InfiniteWidthBNNKernel(ContinuousKernel):
    features: Optional[List[str]] = None
    type: Literal["InfiniteWidthBNNKernel"] = "InfiniteWidthBNNKernel"
    depth: PositiveInt = 3


class SphericalLinearKernel(ContinuousKernel):
    type: Literal["SphericalLinearKernel"] = "SphericalLinearKernel"
    ard: bool = True
    lengthscale_prior: Optional[AnyPrior] = Field(
        default_factory=HVARFNER_LENGTHSCALE_PRIOR
    )
    lengthscale_constraint: Optional[AnyPriorConstraint] = None
    bounds: Union[tuple[float, float], List[tuple[float, float]]] = (0.0, 1.0)

    @model_validator(mode="after")
    def validate_ard_bounds(self):
        if not self.ard and not isinstance(self.bounds, list):
            raise ValueError(
                "Cannot determine number of dimensions. If ard=False then list of bounds should have length equal to the input dimension."
            )
        return self
