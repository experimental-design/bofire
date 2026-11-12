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

    Raises:
        ValueError: If a different strategy with the same ``type`` discriminator
            is already registered.
    """
    import bofire.data_models.strategies.actual_strategy_type as ast_mod
    import bofire.data_models.strategies.api as strategies_api
    from bofire.data_models._register_utils import patch_field, register_into
    from bofire.data_models.strategies.meta_strategy_type import MetaStrategy
    from bofire.data_models.strategies.stepwise.stepwise import Step, StepwiseStrategy

    if not register_into(
        ast_mod._ACTUAL_STRATEGY_TYPES, data_model_cls, kind="strategy"
    ):
        return
    ast_mod.ActualStrategy = tagged_union(*ast_mod._ACTUAL_STRATEGY_TYPES)
    strategies_api.AnyStrategy = tagged_union(
        *ast_mod._ACTUAL_STRATEGY_TYPES, MetaStrategy
    )

    patch_field(Step, "strategy_data", ast_mod.ActualStrategy)
    Step.model_rebuild(force=True)
    StepwiseStrategy.model_rebuild(force=True)


def _rebuild_scalarization_union() -> None:
    """Rebuild ``AnyScalarization`` from the registry and the models using it."""
    import bofire.data_models.strategies.api as strategies_api
    import bofire.data_models.strategies.predictives.scalarization as scalarization_mod
    from bofire.data_models._register_utils import patch_field
    from bofire.data_models.strategies.predictives.multi_fidelity import (
        MultiFidelityVarianceBasedStrategy,
    )
    from bofire.data_models.strategies.predictives.sobo import SoboStrategy
    from bofire.data_models.strategies.stepwise.stepwise import Step, StepwiseStrategy

    scalarization_mod.AnyScalarization = tagged_union(
        *scalarization_mod._SCALARIZATION_TYPES
    )
    strategies_api.AnyScalarization = scalarization_mod.AnyScalarization

    for cls in (SoboStrategy, MultiFidelityVarianceBasedStrategy):
        patch_field(cls, "scalarization", scalarization_mod.AnyScalarization)
        cls.model_rebuild(force=True)
    Step.model_rebuild(force=True)
    StepwiseStrategy.model_rebuild(force=True)


def register_scalarization(data_model_cls: type) -> None:
    """Register a custom scalarization so it is accepted in AnyScalarization fields.

    Rebuilds the ``AnyScalarization`` union with the new type appended, and calls
    ``model_rebuild`` on the strategies that take a scalarization and on the
    models embedding strategies (``Step``, ``StepwiseStrategy``).

    Users must separately register a mapping function via
    ``bofire.strategies.api.register_scalarization`` so that the data model
    can be turned into a BoTorch objective at runtime.

    Args:
        data_model_cls: A concrete subclass of ``Scalarization``.

    Raises:
        ValueError: If a different scalarization with the same ``type``
            discriminator is already registered.
    """
    import bofire.data_models.strategies.predictives.scalarization as scalarization_mod
    from bofire.data_models._register_utils import register_into

    if not register_into(
        scalarization_mod._SCALARIZATION_TYPES, data_model_cls, kind="scalarization"
    ):
        return
    _rebuild_scalarization_union()
