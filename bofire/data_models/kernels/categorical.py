from typing import Literal, Optional

from bofire.data_models.kernels.kernel import FeatureSpecificKernel
from bofire.data_models.priors.api import AnyPrior, AnyPriorConstraint


class CategoricalKernel(FeatureSpecificKernel):
    pass


class HammingDistanceKernel(CategoricalKernel):
    type: Literal["HammingDistanceKernel"] = "HammingDistanceKernel"
    ard: bool = True
    lengthscale_prior: Optional[AnyPrior] = None
    lengthscale_constraint: Optional[AnyPriorConstraint] = None
