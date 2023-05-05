from typing import Union

from bofire.data_models.kernels.aggregation import (
    AdditiveKernel,
    MultiplicativeKernel,
    ScaleKernel,
)
from bofire.data_models.kernels.categorical import (
    CategoricalKernel,
    HammingDistanceKernel,
    TanimotoKernel,
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

AnyCategoricalKernal = Union[HammingDistanceKernel, TanimotoKernel]

AnyKernel = Union[
    AdditiveKernel,
    MultiplicativeKernel,
    ScaleKernel,
    HammingDistanceKernel,
    TanimotoKernel,
    LinearKernel,
    MaternKernel,
    RBFKernel,
]
