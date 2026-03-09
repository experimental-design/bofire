"""Registration utilities for custom engineered feature types."""

import typing
from collections.abc import Sequence
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

    # Lazy import to avoid circular dependencies
    from bofire.data_models.domain.features import EngineeredFeatures

    # Patch the Sequence[Union[...]] annotation on EngineeredFeatures.features
    old = EngineeredFeatures.model_fields["features"].annotation
    inner_args = typing.get_args(typing.get_args(old)[0])
    if data_model_cls not in inner_args:
        new_inner = Union[tuple(list(inner_args) + [data_model_cls])]
        new_ann = Sequence[new_inner]
        EngineeredFeatures.__annotations__["features"] = new_ann
        EngineeredFeatures.model_fields["features"].annotation = new_ann
    EngineeredFeatures.model_rebuild(force=True)
