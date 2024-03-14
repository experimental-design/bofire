from typing import Literal

from bofire.data_models.kernels.kernel import Kernel


class CategoricalKernel(Kernel):
    pass


class HammingDistanceKernel(CategoricalKernel):
    type: Literal["HammingDistanceKernel"] = "HammingDistanceKernel"
    ard: bool = True
