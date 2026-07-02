"""Registration utilities for custom convergence criteria."""

from bofire.data_models.unions import tagged_union


def _all_subclasses(cls: type):
    """Yield all (recursive) subclasses of *cls*."""
    for sub in cls.__subclasses__():
        yield sub
        yield from _all_subclasses(sub)


def _rebuild_dependent_models() -> None:
    """Patch the ``convergence_criterion`` field on all strategy models.

    The convergence criterion is an (optional) field on the base ``Strategy``
    data model and therefore inherited by every concrete strategy. To make a
    newly registered criterion acceptable, the field annotation on the base
    model and all of its subclasses is patched to the rebuilt
    ``AnyConvergenceCriterion`` union and the models are rebuilt.
    """
    import bofire.data_models.strategies.convergence_criteria.api as cc_api
    from bofire.data_models._register_utils import patch_field
    from bofire.data_models.strategies.strategy import Strategy

    for model_cls in [Strategy, *_all_subclasses(Strategy)]:
        if "convergence_criterion" in model_cls.model_fields:
            patch_field(
                model_cls,
                "convergence_criterion",
                cc_api.AnyConvergenceCriterion,
            )
            model_cls.model_rebuild(force=True)


def register_convergence_criterion(data_model_cls: type) -> None:
    """Register a custom convergence criterion data model.

    This appends the type to the internal registry, rebuilds the
    ``AnyConvergenceCriterion`` union, and patches/rebuilds the
    ``convergence_criterion`` field on the base ``Strategy`` and all of its
    subclasses so that Pydantic accepts the new type.

    Args:
        data_model_cls: A concrete subclass of ``ConvergenceCriterion``.

    Raises:
        ValueError: If a different convergence criterion with the same ``type``
            discriminator is already registered.
    """
    import bofire.data_models.strategies.convergence_criteria.api as cc_api
    from bofire.data_models._register_utils import register_into

    if not register_into(
        cc_api._CONVERGENCE_CRITERION_TYPES,
        data_model_cls,
        kind="convergence criterion",
    ):
        return
    cc_api.AnyConvergenceCriterion = tagged_union(*cc_api._CONVERGENCE_CRITERION_TYPES)
    _rebuild_dependent_models()
