from typing import Union

from bofire.data_models.kernels.aggregation import (
    AdditiveKernel,
    MultiplicativeKernel,
    ScaleKernel,
)
from bofire.data_models.kernels.categorical import (
    CategoricalKernel,
    HammondDistanceKernel,
)
from bofire.data_models.kernels.continuous import (
    ContinuousKernel,
    LinearKernel,
    MaternKernel,
    RBFKernel,
)
from bofire.data_models.kernels.kernel import Kernel

AbstractKernel = Union[
    Kernel,
    CategoricalKernel,
    ContinuousKernel,
]

AnyContinuousKernel = Union[
    MaternKernel,
    LinearKernel,
    RBFKernel,
]

AnyCategoricalKernal = HammondDistanceKernel

AnyKernel = Union[
    AdditiveKernel,
    MultiplicativeKernel,
    ScaleKernel,
    HammondDistanceKernel,
    LinearKernel,
    MaternKernel,
    RBFKernel,
]
