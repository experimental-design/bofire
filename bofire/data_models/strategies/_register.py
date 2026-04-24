"""Registration utilities for custom strategy types."""

from bofire.data_models.unions import tagged_union


def register_strategy(data_model_cls: type) -> None:
    """Register a custom strategy type so it is accepted in ActualStrategy fields.

    This appends the type to the internal registry, rebuilds the
    ``ActualStrategy`` union, and calls ``model_rebuild`` on the
    ``Step`` and ``StepwiseStrategy`` models so that Pydantic accepts the
    new type.

    Args:
        data_model_cls: A concrete subclass of ``Strategy``.
    """
    import bofire.data_models.strategies.actual_strategy_type as ast_mod
    import bofire.data_models.strategies.api as strategies_api
    from bofire.data_models._register_utils import patch_field
    from bofire.data_models.strategies.meta_strategy_type import MetaStrategy
    from bofire.data_models.strategies.stepwise.stepwise import Step, StepwiseStrategy

    if data_model_cls in ast_mod._ACTUAL_STRATEGY_TYPES:
        return
    ast_mod._ACTUAL_STRATEGY_TYPES.append(data_model_cls)
    ast_mod.ActualStrategy = tagged_union(*ast_mod._ACTUAL_STRATEGY_TYPES)
    strategies_api.AnyStrategy = tagged_union(
        *ast_mod._ACTUAL_STRATEGY_TYPES, MetaStrategy
    )

    patch_field(Step, "strategy_data", ast_mod.ActualStrategy)
    Step.model_rebuild(force=True)
    StepwiseStrategy.model_rebuild(force=True)
