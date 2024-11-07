from typing import Union

from bofire.data_models.kernels.aggregation import (
    AdditiveKernel,
    MultiplicativeKernel,
    ScaleKernel,
)
from bofire.data_models.kernels.categorical import (
    CategoricalKernel,
    HammingDistanceKernel,
)
from bofire.data_models.kernels.continuous import (
    ContinuousKernel,
    InfiniteWidthBNNKernel,
    LinearKernel,
    MaternKernel,
    PolynomialKernel,
    RBFKernel,
)
from bofire.data_models.kernels.kernel import Kernel
from bofire.data_models.kernels.molecular import MolecularKernel, TanimotoKernel
from bofire.data_models.kernels.shape import WassersteinKernel


AbstractKernel = Union[Kernel, CategoricalKernel, ContinuousKernel, MolecularKernel]

AnyContinuousKernel = Union[
    MaternKernel,
    LinearKernel,
    PolynomialKernel,
    RBFKernel,
    InfiniteWidthBNNKernel,
]

AnyCategoricalKernel = HammingDistanceKernel

AnyMolecularKernel = TanimotoKernel

AnyKernel = Union[
    AdditiveKernel,
    MultiplicativeKernel,
    ScaleKernel,
    HammingDistanceKernel,
    LinearKernel,
    PolynomialKernel,
    MaternKernel,
    RBFKernel,
    TanimotoKernel,
    InfiniteWidthBNNKernel,
    WassersteinKernel,
]
