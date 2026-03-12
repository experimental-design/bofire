"""Registration utilities for custom engineered feature types."""

from typing import Union


def register_engineered_feature(data_model_cls: type) -> None:
    """Register a custom engineered feature type so it is accepted in EngineeredFeatures.

    This appends the type to the internal registry, rebuilds the
    ``AnyEngineeredFeature`` union, and calls ``model_rebuild`` on
    ``EngineeredFeatures`` so that Pydantic accepts the new type.

    Args:
        data_model_cls: A concrete subclass of ``EngineeredFeature``.
    """
    import bofire.data_models.features.api as features_api

    if data_model_cls in features_api._ENGINEERED_FEATURE_TYPES:
        return
    features_api._ENGINEERED_FEATURE_TYPES.append(data_model_cls)
    features_api.AnyEngineeredFeature = Union[
        tuple(features_api._ENGINEERED_FEATURE_TYPES)
    ]

    from bofire.data_models._register_utils import append_to_union_field
    from bofire.data_models.domain.features import EngineeredFeatures

    append_to_union_field(EngineeredFeatures, "features", data_model_cls)
    EngineeredFeatures.model_rebuild(force=True)
