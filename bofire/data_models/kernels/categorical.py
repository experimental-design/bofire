from typing import Literal

from bofire.data_models.kernels.kernel import Kernel


class CategoricalKernel(Kernel):
    pass


class HammondDistanceKernel(CategoricalKernel):
    type: Literal["HammondDistanceKernel"] = "HammondDistanceKernel"
    ard: bool = True
