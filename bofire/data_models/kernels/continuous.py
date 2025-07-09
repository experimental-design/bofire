from typing import List, Literal, Optional

from pydantic import PositiveInt, field_validator

from bofire.data_models.kernels.kernel import FeatureSpecificKernel
from bofire.data_models.priors.api import AnyGeneralPrior, AnyPrior, AnyPriorConstraint


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
