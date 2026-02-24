from typing import List, Type, Union

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
from bofire.data_models.kernels.conditional import (
    ConditionalEmbeddingKernel,
    WedgeKernel,
)
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

_CONTINUOUS_KERNEL_TYPES: List[Type[ContinuousKernel]] = [
    MaternKernel,
    LinearKernel,
    PolynomialKernel,
    RBFKernel,
    SphericalLinearKernel,
    InfiniteWidthBNNKernel,
]

AnyContinuousKernel = Union[tuple(_CONTINUOUS_KERNEL_TYPES)]

_CATEGORICAL_KERNEL_TYPES: List[Type[CategoricalKernel]] = [
    HammingDistanceKernel,
    IndexKernel,
    PositiveIndexKernel,
]

AnyCategoricalKernel = Union[tuple(_CATEGORICAL_KERNEL_TYPES)]

AnyMolecularKernel = TanimotoKernel

_KERNEL_TYPES: List[Type[Kernel]] = [
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


def _rebuild_dependent_models(new_kernel_cls: Type[Kernel]) -> None:
    """Rebuild all Pydantic models whose fields reference kernel unions."""
    from bofire.data_models._register_utils import append_to_union_field, patch_field
    from bofire.data_models.surrogates.botorch_surrogates import BotorchSurrogates
    from bofire.data_models.surrogates.mixed_single_task_gp import (
        MixedSingleTaskGPSurrogate,
    )
    from bofire.data_models.surrogates.multi_task_gp import MultiTaskGPSurrogate
    from bofire.data_models.surrogates.single_task_gp import SingleTaskGPSurrogate
    from bofire.data_models.surrogates.tanimoto_gp import TanimotoGPSurrogate

    # Add new kernel type to aggregation kernel inline unions
    # (handles both Sequence[Union[...]] and plain Union[...])
    for model_cls, field_name in [
        (AdditiveKernel, "kernels"),
        (MultiplicativeKernel, "kernels"),
        (PolynomialFeatureInteractionKernel, "kernels"),
        (ScaleKernel, "base_kernel"),
        (ConditionalEmbeddingKernel, "base_kernel"),
        (WedgeKernel, "base_kernel"),
    ]:
        append_to_union_field(model_cls, field_name, new_kernel_cls)

    # Rebuild aggregation and conditional kernels
    for cls in [
        AdditiveKernel,
        MultiplicativeKernel,
        ScaleKernel,
        PolynomialFeatureInteractionKernel,
        ConditionalEmbeddingKernel,
        WedgeKernel,
    ]:
        cls.model_rebuild(force=True)

    # Patch AnyKernel fields on surrogate models
    for model_cls, field_name in [
        (SingleTaskGPSurrogate, "kernel"),
        (MultiTaskGPSurrogate, "kernel"),
        (TanimotoGPSurrogate, "kernel"),
    ]:
        patch_field(model_cls, field_name, AnyKernel)

    # Patch sub-category kernel fields if the new type is a subclass
    if issubclass(new_kernel_cls, ContinuousKernel):
        patch_field(
            MixedSingleTaskGPSurrogate, "continuous_kernel", AnyContinuousKernel
        )
    if issubclass(new_kernel_cls, CategoricalKernel):
        patch_field(
            MixedSingleTaskGPSurrogate, "categorical_kernel", AnyCategoricalKernel
        )

    # Rebuild surrogate models
    for cls in [
        SingleTaskGPSurrogate,
        MultiTaskGPSurrogate,
        TanimotoGPSurrogate,
        MixedSingleTaskGPSurrogate,
    ]:
        cls.model_rebuild(force=True)

    # Rebuild BotorchSurrogates
    BotorchSurrogates.model_rebuild(force=True)


def register_kernel(data_model_cls: Type[Kernel]) -> None:
    """Register a custom kernel type so it is accepted in AnyKernel fields.

    This appends the type to the internal registry, rebuilds the
    ``AnyKernel`` union, and calls ``model_rebuild`` on all dependent
    Pydantic models (aggregation kernels, surrogates) so that the new
    type is accepted.

    If the type is a subclass of ``ContinuousKernel`` or ``CategoricalKernel``,
    it is also added to the corresponding sub-category union
    (``AnyContinuousKernel`` / ``AnyCategoricalKernel``).

    Args:
        data_model_cls: A concrete subclass of ``Kernel``.
    """
    global AnyKernel, AnyContinuousKernel, AnyCategoricalKernel
    if data_model_cls in _KERNEL_TYPES:
        return
    _KERNEL_TYPES.append(data_model_cls)
    AnyKernel = Union[tuple(_KERNEL_TYPES)]

    # Auto-detect sub-category from base class
    if issubclass(data_model_cls, ContinuousKernel):
        _CONTINUOUS_KERNEL_TYPES.append(data_model_cls)
        AnyContinuousKernel = Union[tuple(_CONTINUOUS_KERNEL_TYPES)]
    elif issubclass(data_model_cls, CategoricalKernel):
        _CATEGORICAL_KERNEL_TYPES.append(data_model_cls)
        AnyCategoricalKernel = Union[tuple(_CATEGORICAL_KERNEL_TYPES)]

    _rebuild_dependent_models(data_model_cls)
