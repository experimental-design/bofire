from typing import Literal

from bofire.data_models.kernels.kernel import ConcreteKernel


class CategoricalKernel(ConcreteKernel):
    pass


class HammingDistanceKernel(CategoricalKernel):
    type: Literal["HammingDistanceKernel"] = "HammingDistanceKernel"
    ard: bool = True
