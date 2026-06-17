"""Registration utilities for custom prior and prior constraint types."""

from bofire.data_models.unions import tagged_union


def _rebuild_dependent_models() -> None:
    """Rebuild all Pydantic models whose fields reference AnyPrior or AnyPriorConstraint."""
    import bofire.data_models.priors.api as priors_api
    from bofire.data_models._register_utils import patch_field

    # Lazy imports to avoid circular dependencies
    from bofire.data_models.kernels.aggregation import (
        AdditiveKernel,
        MultiplicativeKernel,
        PolynomialFeatureInteractionKernel,
        ScaleKernel,
    )
    from bofire.data_models.kernels.categorical import (
        HammingDistanceKernel,
        IndexKernel,
        PositiveIndexKernel,
    )
    from bofire.data_models.kernels.conditional import WedgeKernel
    from bofire.data_models.kernels.continuous import (
        LinearKernel,
        MaternKernel,
        PolynomialKernel,
        RBFKernel,
        SphericalLinearKernel,
    )
    from bofire.data_models.kernels.shape import (
        ExactWassersteinKernel,
        WassersteinKernel,
    )
    from bofire.data_models.surrogates.botorch_surrogates import BotorchSurrogates
    from bofire.data_models.surrogates.linear import LinearSurrogate
    from bofire.data_models.surrogates.mixed_single_task_gp import (
        MixedSingleTaskGPSurrogate,
    )
    from bofire.data_models.surrogates.multi_task_gp import MultiTaskGPSurrogate
    from bofire.data_models.surrogates.polynomial import PolynomialSurrogate
    from bofire.data_models.surrogates.robust_single_task_gp import (
        RobustSingleTaskGPSurrogate,
    )
    from bofire.data_models.surrogates.single_task_gp import (
        SingleTaskGPHyperconfig,
        SingleTaskGPSurrogate,
    )
    from bofire.data_models.surrogates.tanimoto_gp import TanimotoGPSurrogate

    AnyPrior = priors_api.AnyPrior
    AnyPriorConstraint = priors_api.AnyPriorConstraint

    # Patch AnyPrior fields
    for model_cls, field_name in [
        (RBFKernel, "lengthscale_prior"),
        (MaternKernel, "lengthscale_prior"),
        (SphericalLinearKernel, "lengthscale_prior"),
        (HammingDistanceKernel, "lengthscale_prior"),
        (IndexKernel, "prior"),
        (PositiveIndexKernel, "prior"),
        (PositiveIndexKernel, "task_prior"),
        (PositiveIndexKernel, "diag_prior"),
        (WedgeKernel, "lengthscale_prior"),
        (WedgeKernel, "angle_prior"),
        (WedgeKernel, "radius_prior"),
        (ScaleKernel, "outputscale_prior"),
        (LinearKernel, "variance_prior"),
        (PolynomialKernel, "offset_prior"),
        (PolynomialFeatureInteractionKernel, "outputscale_prior"),
        (WassersteinKernel, "lengthscale_prior"),
        (ExactWassersteinKernel, "lengthscale_prior"),
        (SingleTaskGPSurrogate, "noise_prior"),
        (MultiTaskGPSurrogate, "noise_prior"),
        (MixedSingleTaskGPSurrogate, "noise_prior"),
        (TanimotoGPSurrogate, "noise_prior"),
        (PolynomialSurrogate, "noise_prior"),
        (LinearSurrogate, "noise_prior"),
        (RobustSingleTaskGPSurrogate, "noise_prior"),
    ]:
        patch_field(model_cls, field_name, AnyPrior)

    # Patch AnyPriorConstraint fields
    for model_cls, field_name in [
        (RBFKernel, "lengthscale_constraint"),
        (MaternKernel, "lengthscale_constraint"),
        (SphericalLinearKernel, "lengthscale_constraint"),
        (HammingDistanceKernel, "lengthscale_constraint"),
        (IndexKernel, "var_constraint"),
        (PositiveIndexKernel, "var_constraint"),
        (WedgeKernel, "lengthscale_constraint"),
        (ScaleKernel, "outputscale_constraint"),
        (SingleTaskGPSurrogate, "noise_constraint"),
        (MultiTaskGPSurrogate, "noise_constraint"),
        (MixedSingleTaskGPSurrogate, "noise_constraint"),
        (TanimotoGPSurrogate, "noise_constraint"),
        (PolynomialSurrogate, "noise_constraint"),
        (LinearSurrogate, "noise_constraint"),
        (RobustSingleTaskGPSurrogate, "noise_constraint"),
        (SingleTaskGPHyperconfig, "lengthscale_constraint"),
        (SingleTaskGPHyperconfig, "outputscale_constraint"),
    ]:
        patch_field(model_cls, field_name, AnyPriorConstraint)

    # Rebuild in dependency order:
    # 1. Leaf kernel models
    for cls in [
        RBFKernel,
        MaternKernel,
        LinearKernel,
        PolynomialKernel,
        SphericalLinearKernel,
        HammingDistanceKernel,
        IndexKernel,
        PositiveIndexKernel,
        WassersteinKernel,
        ExactWassersteinKernel,
    ]:
        cls.model_rebuild(force=True)

    # 2. Conditional kernels (reference leaf kernels)
    WedgeKernel.model_rebuild(force=True)

    # 3. Aggregation kernels (embed leaf kernel types)
    for cls in [
        AdditiveKernel,
        MultiplicativeKernel,
        ScaleKernel,
        PolynomialFeatureInteractionKernel,
    ]:
        cls.model_rebuild(force=True)

    # 4. Surrogate models
    SingleTaskGPHyperconfig.model_rebuild(force=True)
    for cls in [
        SingleTaskGPSurrogate,
        MultiTaskGPSurrogate,
        MixedSingleTaskGPSurrogate,
        TanimotoGPSurrogate,
        PolynomialSurrogate,
        LinearSurrogate,
        RobustSingleTaskGPSurrogate,
    ]:
        cls.model_rebuild(force=True)

    # 5. BotorchSurrogates
    BotorchSurrogates.model_rebuild(force=True)


def register_prior(data_model_cls: type, overwrite: bool = False) -> None:
    """Register a custom prior type so it is accepted in AnyPrior fields.

    This appends the type to the internal registry, rebuilds the
    ``AnyPrior`` union, and calls ``model_rebuild`` on all dependent
    Pydantic models (kernels, surrogates) so that the new type is accepted.

    Args:
        data_model_cls: A concrete subclass of ``Prior``.
        overwrite: If ``True``, replace an existing prior registered under the
            same ``type`` discriminator instead of raising.

    Raises:
        ValueError: If a different prior with the same ``type`` is already
            registered and *overwrite* is ``False``.
    """
    import bofire.data_models.priors.api as priors_api
    from bofire.data_models._register_utils import register_into

    action, _ = register_into(
        priors_api._PRIOR_TYPES, data_model_cls, overwrite=overwrite, kind="prior"
    )
    if action == "noop":
        return
    priors_api.AnyPrior = tagged_union(*priors_api._PRIOR_TYPES)
    _rebuild_dependent_models()


def register_prior_constraint(data_model_cls: type, overwrite: bool = False) -> None:
    """Register a custom prior constraint type so it is accepted in AnyPriorConstraint fields.

    This appends the type to the internal registry, rebuilds the
    ``AnyPriorConstraint`` union, and calls ``model_rebuild`` on all
    dependent Pydantic models.

    Args:
        data_model_cls: A concrete subclass of ``PriorConstraint`` or ``Interval``.
        overwrite: If ``True``, replace an existing prior constraint registered
            under the same ``type`` discriminator instead of raising.

    Raises:
        ValueError: If a different prior constraint with the same ``type`` is
            already registered and *overwrite* is ``False``.
    """
    import bofire.data_models.priors.api as priors_api
    from bofire.data_models._register_utils import register_into

    action, _ = register_into(
        priors_api._PRIOR_CONSTRAINT_TYPES,
        data_model_cls,
        overwrite=overwrite,
        kind="prior constraint",
    )
    if action == "noop":
        return
    priors_api.AnyPriorConstraint = tagged_union(*priors_api._PRIOR_CONSTRAINT_TYPES)
    _rebuild_dependent_models()
