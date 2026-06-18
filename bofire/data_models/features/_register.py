"""Registration utilities for custom engineered feature types."""

from bofire.data_models.unions import tagged_union


def register_engineered_feature(data_model_cls: type, overwrite: bool = False) -> None:
    """Register a custom engineered feature type so it is accepted in EngineeredFeatures.

    This appends the type to the internal registry, rebuilds the
    ``AnyEngineeredFeature`` discriminated union, and calls ``model_rebuild``
    on ``EngineeredFeatures`` so that Pydantic accepts the new type.

    Args:
        data_model_cls: A concrete subclass of ``EngineeredFeature``.
        overwrite: If ``True``, replace an existing engineered feature
            registered under the same ``type`` discriminator instead of raising.

    Raises:
        ValueError: If a different engineered feature with the same ``type`` is
            already registered and *overwrite* is ``False``.
    """
    import bofire.data_models.features.api as features_api
    from bofire.data_models._register_utils import append_to_union_field, register_into
    from bofire.data_models.domain.features import EngineeredFeatures

    changed, _ = register_into(
        features_api._ENGINEERED_FEATURE_TYPES,
        data_model_cls,
        overwrite=overwrite,
        kind="engineered feature",
    )
    if not changed:
        return
    features_api.AnyEngineeredFeature = tagged_union(
        *features_api._ENGINEERED_FEATURE_TYPES
    )

    append_to_union_field(EngineeredFeatures, "features", data_model_cls)
    EngineeredFeatures.model_rebuild(force=True)
