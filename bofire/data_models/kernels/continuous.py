from typing import Literal, Optional

from bofire.data_models.kernels.kernel import Kernel
from bofire.data_models.priors.api import AnyPrior


class ContinuousKernel(Kernel):
    pass


class RBFKernel(ContinuousKernel):
    type: Literal["RBFKernel"] = "RBFKernel"
    ard: bool = True
    lengthscale_prior: Optional[AnyPrior] = None


class MaternKernel(ContinuousKernel):
    type: Literal["MaternKernel"] = "MaternKernel"
    ard: bool = True
    nu: float = 2.5
    lengthscale_prior: Optional[AnyPrior] = None


class LinearKernel(ContinuousKernel):
    type: Literal["LinearKernel"] = "LinearKernel"
    variance_prior: Optional[AnyPrior] = None


class PolynomialKernel(ContinuousKernel):
    type: Literal["PolynomialKernel"] = "PolynomialKernel"
    offset_prior: Optional[AnyPrior] = None
    power: int = 2
