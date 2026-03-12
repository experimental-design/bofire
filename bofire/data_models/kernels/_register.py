"""Registration utilities for custom kernel types."""

from typing import Union


def _rebuild_dependent_models(new_kernel_cls: type) -> None:
    """Rebuild all Pydantic models whose fields reference kernel unions."""
    import bofire.data_models.kernels.api as kernels_api
    from bofire.data_models._register_utils import append_to_union_field, patch_field
    from bofire.data_models.kernels.aggregation import (
        AdditiveKernel,
        MultiplicativeKernel,
        PolynomialFeatureInteractionKernel,
        ScaleKernel,
    )
    from bofire.data_models.kernels.categorical import CategoricalKernel
    from bofire.data_models.kernels.conditional import (
        ConditionalEmbeddingKernel,
        WedgeKernel,
    )
    from bofire.data_models.kernels.continuous import ContinuousKernel
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
        patch_field(model_cls, field_name, kernels_api.AnyKernel)

    # Patch sub-category kernel fields if the new type is a subclass
    if issubclass(new_kernel_cls, ContinuousKernel):
        patch_field(
            MixedSingleTaskGPSurrogate,
            "continuous_kernel",
            kernels_api.AnyContinuousKernel,
        )
    if issubclass(new_kernel_cls, CategoricalKernel):
        patch_field(
            MixedSingleTaskGPSurrogate,
            "categorical_kernel",
            kernels_api.AnyCategoricalKernel,
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


def register_kernel(data_model_cls: type) -> None:
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
    import bofire.data_models.kernels.api as kernels_api
    from bofire.data_models.kernels.categorical import CategoricalKernel
    from bofire.data_models.kernels.continuous import ContinuousKernel

    if data_model_cls in kernels_api._KERNEL_TYPES:
        return
    kernels_api._KERNEL_TYPES.append(data_model_cls)
    kernels_api.AnyKernel = Union[tuple(kernels_api._KERNEL_TYPES)]

    # Auto-detect sub-category from base class
    if issubclass(data_model_cls, ContinuousKernel):
        kernels_api._CONTINUOUS_KERNEL_TYPES.append(data_model_cls)
        kernels_api.AnyContinuousKernel = Union[
            tuple(kernels_api._CONTINUOUS_KERNEL_TYPES)
        ]
    elif issubclass(data_model_cls, CategoricalKernel):
        kernels_api._CATEGORICAL_KERNEL_TYPES.append(data_model_cls)
        kernels_api.AnyCategoricalKernel = Union[
            tuple(kernels_api._CATEGORICAL_KERNEL_TYPES)
        ]

    _rebuild_dependent_models(data_model_cls)
