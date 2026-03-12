from typing import Union

from bofire.data_models.kernels._register import register_kernel  # noqa: F401
from bofire.data_models.kernels.aggregation import (
    AdditiveKernel,
    MultiplicativeKernel,
    PolynomialFeatureInteractionKernel,
    ScaleKernel,
)
from bofire.data_models.kernels.categorical import (
    CategoricalKernel,
    HammingDistanceKernel,
    IndexKernel,
    PositiveIndexKernel,
)
from bofire.data_models.kernels.conditional import WedgeKernel
from bofire.data_models.kernels.continuous import (
    ContinuousKernel,
    InfiniteWidthBNNKernel,
    LinearKernel,
    MaternKernel,
    PolynomialKernel,
    RBFKernel,
    SphericalLinearKernel,
)
from bofire.data_models.kernels.kernel import (
    AggregationKernel,
    FeatureSpecificKernel,
    Kernel,
)
from bofire.data_models.kernels.molecular import MolecularKernel, TanimotoKernel
from bofire.data_models.kernels.shape import WassersteinKernel


AbstractKernel = Union[
    Kernel,
    CategoricalKernel,
    ContinuousKernel,
    MolecularKernel,
    FeatureSpecificKernel,
    AggregationKernel,
]

_CONTINUOUS_KERNEL_TYPES: list[type[ContinuousKernel]] = [
    MaternKernel,
    LinearKernel,
    PolynomialKernel,
    RBFKernel,
    SphericalLinearKernel,
    InfiniteWidthBNNKernel,
]

AnyContinuousKernel = Union[tuple(_CONTINUOUS_KERNEL_TYPES)]

_CATEGORICAL_KERNEL_TYPES: list[type[CategoricalKernel]] = [
    HammingDistanceKernel,
    IndexKernel,
    PositiveIndexKernel,
]

AnyCategoricalKernel = Union[tuple(_CATEGORICAL_KERNEL_TYPES)]

AnyMolecularKernel = TanimotoKernel

_KERNEL_TYPES: list[type[Kernel]] = [
    AdditiveKernel,
    MultiplicativeKernel,
    PolynomialFeatureInteractionKernel,
    ScaleKernel,
    HammingDistanceKernel,
    IndexKernel,
    PositiveIndexKernel,
    LinearKernel,
    PolynomialKernel,
    MaternKernel,
    RBFKernel,
    SphericalLinearKernel,
    TanimotoKernel,
    InfiniteWidthBNNKernel,
    WassersteinKernel,
    WedgeKernel,
]

AnyKernel = Union[tuple(_KERNEL_TYPES)]
