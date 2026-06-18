"""Registration utilities for custom strategy types."""

from bofire.data_models.unions import tagged_union


def register_strategy(data_model_cls: type, overwrite: bool = False) -> None:
    """Register a custom strategy type so it is accepted in ActualStrategy fields.

    This appends the type to the internal registry, rebuilds the
    ``ActualStrategy`` union, and calls ``model_rebuild`` on the
    ``Step`` and ``StepwiseStrategy`` models so that Pydantic accepts the
    new type.

    Args:
        data_model_cls: A concrete subclass of ``Strategy``.
        overwrite: If ``True``, replace an existing strategy registered under
            the same ``type`` discriminator instead of raising. Useful when
            re-running code that re-defines and re-registers the same strategy.

    Raises:
        ValueError: If a different strategy with the same ``type`` is already
            registered and *overwrite* is ``False``.
    """
    import bofire.data_models.strategies.actual_strategy_type as ast_mod
    import bofire.data_models.strategies.api as strategies_api
    from bofire.data_models._register_utils import patch_field, register_into
    from bofire.data_models.strategies.meta_strategy_type import MetaStrategy
    from bofire.data_models.strategies.stepwise.stepwise import Step, StepwiseStrategy

    changed, _ = register_into(
        ast_mod._ACTUAL_STRATEGY_TYPES,
        data_model_cls,
        overwrite=overwrite,
        kind="strategy",
    )
    if not changed:
        return
    ast_mod.ActualStrategy = tagged_union(*ast_mod._ACTUAL_STRATEGY_TYPES)
    strategies_api.AnyStrategy = tagged_union(
        *ast_mod._ACTUAL_STRATEGY_TYPES, MetaStrategy
    )

    patch_field(Step, "strategy_data", ast_mod.ActualStrategy)
    Step.model_rebuild(force=True)
    StepwiseStrategy.model_rebuild(force=True)
