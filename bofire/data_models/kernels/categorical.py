from typing import Literal

from bofire.data_models.kernels.kernel import FeatureSpecificKernel


class CategoricalKernel(FeatureSpecificKernel):
    pass


class HammingDistanceKernel(CategoricalKernel):
    type: Literal["HammingDistanceKernel"] = "HammingDistanceKernel"
    ard: bool = True
