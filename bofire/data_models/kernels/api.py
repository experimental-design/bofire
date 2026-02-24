import typing
from collections.abc import Sequence
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

AnyContinuousKernel = Union[
    MaternKernel,
    LinearKernel,
    PolynomialKernel,
    RBFKernel,
    SphericalLinearKernel,
    InfiniteWidthBNNKernel,
]

AnyCategoricalKernel = Union[
    HammingDistanceKernel,
    IndexKernel,
    PositiveIndexKernel,
]

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
    """Rebuild all Pydantic models whose fields reference AnyKernel."""
    from bofire.data_models.surrogates.botorch_surrogates import BotorchSurrogates
    from bofire.data_models.surrogates.multi_task_gp import MultiTaskGPSurrogate
    from bofire.data_models.surrogates.single_task_gp import SingleTaskGPSurrogate
    from bofire.data_models.surrogates.tanimoto_gp import TanimotoGPSurrogate

    # Add new kernel type to aggregation kernel inline unions
    for model_cls, field_name in [
        (AdditiveKernel, "kernels"),
        (MultiplicativeKernel, "kernels"),
        (PolynomialFeatureInteractionKernel, "kernels"),
    ]:
        old = model_cls.model_fields[field_name].annotation
        # Annotation is Sequence[Union[...]]
        inner = typing.get_args(old)[0]
        inner_args = typing.get_args(inner)
        if new_kernel_cls not in inner_args:
            new_inner = Union[tuple(list(inner_args) + [new_kernel_cls])]
            new_ann = Sequence[new_inner]
            model_cls.__annotations__[field_name] = new_ann
            model_cls.model_fields[field_name].annotation = new_ann

    # ScaleKernel.base_kernel is Union[...] (not Sequence)
    old = ScaleKernel.model_fields["base_kernel"].annotation
    args = typing.get_args(old)
    if new_kernel_cls not in args:
        new_ann = Union[tuple(list(args) + [new_kernel_cls])]
        ScaleKernel.__annotations__["base_kernel"] = new_ann
        ScaleKernel.model_fields["base_kernel"].annotation = new_ann

    # Rebuild aggregation kernels
    for cls in [
        AdditiveKernel,
        MultiplicativeKernel,
        ScaleKernel,
        PolynomialFeatureInteractionKernel,
    ]:
        cls.model_rebuild(force=True)

    # Patch AnyKernel fields on surrogate models
    any_kernel = AnyKernel
    for model_cls, field_name in [
        (SingleTaskGPSurrogate, "kernel"),
        (MultiTaskGPSurrogate, "kernel"),
        (TanimotoGPSurrogate, "kernel"),
    ]:
        model_cls.__annotations__[field_name] = any_kernel
        model_cls.model_fields[field_name].annotation = any_kernel

    # Rebuild surrogate models
    for cls in [SingleTaskGPSurrogate, MultiTaskGPSurrogate, TanimotoGPSurrogate]:
        cls.model_rebuild(force=True)

    # Rebuild BotorchSurrogates
    BotorchSurrogates.model_rebuild(force=True)


def register_kernel(data_model_cls: Type[Kernel]) -> None:
    """Register a custom kernel type so it is accepted in AnyKernel fields.

    This appends the type to the internal registry, rebuilds the
    ``AnyKernel`` union, and calls ``model_rebuild`` on all dependent
    Pydantic models (aggregation kernels, surrogates) so that the new
    type is accepted.

    Args:
        data_model_cls: A concrete subclass of ``Kernel``.
    """
    global AnyKernel
    if data_model_cls in _KERNEL_TYPES:
        return
    _KERNEL_TYPES.append(data_model_cls)
    AnyKernel = Union[tuple(_KERNEL_TYPES)]
    _rebuild_dependent_models(data_model_cls)
