"""Registration utilities for custom likelihood types."""

from bofire.data_models.unions import tagged_union


def _rebuild_dependent_models() -> None:
    """Rebuild all Pydantic models whose fields reference likelihood unions."""
    import bofire.data_models.likelihoods.api as likelihoods_api
    from bofire.data_models._register_utils import patch_field
    from bofire.data_models.surrogates.botorch_surrogates import BotorchSurrogates
    from bofire.data_models.surrogates.multi_task_gp import MultiTaskGPSurrogate
    from bofire.data_models.surrogates.single_task_gp import SingleTaskGPSurrogate

    for model_cls, union in [
        (SingleTaskGPSurrogate, likelihoods_api.AnySingleTaskLikelihood),
        (MultiTaskGPSurrogate, likelihoods_api.AnyMultiTaskLikelihood),
    ]:
        patch_field(model_cls, "likelihood", union)
        model_cls.model_rebuild(force=True)

    BotorchSurrogates.model_rebuild(force=True)


def register_likelihood(
    data_model_cls: type,
    *,
    single_task: bool = True,
    multi_task: bool = False,
) -> None:
    """Register a custom likelihood type so that surrogates accept it.

    The type always joins ``AnyLikelihood``. The flags choose which Gaussian
    processes accept it as their ``likelihood``. Calling this again for a class
    that is already registered adds it to any further GPs its flags name.

    Args:
        data_model_cls: A concrete subclass of ``Likelihood``.
        single_task: Whether a ``SingleTaskGPSurrogate`` accepts it
            (``AnySingleTaskLikelihood``).
        multi_task: Whether a ``MultiTaskGPSurrogate`` accepts it
            (``AnyMultiTaskLikelihood``). Its mapper is then also given the
            ``num_tasks`` and the ``task_feature_index`` of the multi-task GP.

    Raises:
        ValueError: If a different likelihood with the same ``type``
            discriminator is already registered.
    """
    import bofire.data_models.likelihoods.api as likelihoods_api
    from bofire.data_models._register_utils import register_into

    changed = register_into(
        likelihoods_api._LIKELIHOOD_TYPES, data_model_cls, kind="likelihood"
    )
    if changed:
        likelihoods_api.AnyLikelihood = tagged_union(*likelihoods_api._LIKELIHOOD_TYPES)

    if (
        single_task
        and data_model_cls not in likelihoods_api._SINGLE_TASK_LIKELIHOOD_TYPES
    ):
        likelihoods_api._SINGLE_TASK_LIKELIHOOD_TYPES.append(data_model_cls)
        likelihoods_api.AnySingleTaskLikelihood = tagged_union(
            *likelihoods_api._SINGLE_TASK_LIKELIHOOD_TYPES
        )
        changed = True

    if (
        multi_task
        and data_model_cls not in likelihoods_api._MULTI_TASK_LIKELIHOOD_TYPES
    ):
        likelihoods_api._MULTI_TASK_LIKELIHOOD_TYPES.append(data_model_cls)
        likelihoods_api.AnyMultiTaskLikelihood = tagged_union(
            *likelihoods_api._MULTI_TASK_LIKELIHOOD_TYPES
        )
        changed = True

    if changed:
        _rebuild_dependent_models()
