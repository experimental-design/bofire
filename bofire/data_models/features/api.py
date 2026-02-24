import typing
from collections.abc import Sequence
from typing import List, Type, Union

from bofire.data_models.features.categorical import CategoricalInput, CategoricalOutput
from bofire.data_models.features.continuous import ContinuousInput, ContinuousOutput
from bofire.data_models.features.descriptor import (
    CategoricalDescriptorInput,
    ContinuousDescriptorInput,
)
from bofire.data_models.features.discrete import DiscreteInput
from bofire.data_models.features.engineered_feature import (
    CloneFeature,
    EngineeredFeature,
    InterpolateFeature,
    MeanFeature,
    MolecularWeightedMeanFeature,
    MolecularWeightedSumFeature,
    ProductFeature,
    SumFeature,
    WeightedMeanFeature,
    WeightedSumFeature,
)
from bofire.data_models.features.feature import Feature, Input, Output
from bofire.data_models.features.molecular import (
    CategoricalMolecularInput,
    ContinuousMolecularInput,
)
from bofire.data_models.features.numerical import NumericalInput
from bofire.data_models.features.task import TaskInput


AbstractFeature = Union[
    Feature,
    Input,
    Output,
    NumericalInput,
]

# TODO: here is a bug, CategoricalOutput has to be the first item here, no idea why ...
AnyFeature = Union[
    CategoricalOutput,
    DiscreteInput,
    CategoricalInput,
    ContinuousOutput,
    ContinuousInput,
    ContinuousDescriptorInput,
    CategoricalDescriptorInput,
    CategoricalMolecularInput,
    TaskInput,
    SumFeature,
    MeanFeature,
    WeightedMeanFeature,
    WeightedSumFeature,
    MolecularWeightedMeanFeature,
    MolecularWeightedSumFeature,
    ContinuousMolecularInput,
    ProductFeature,
    InterpolateFeature,
    CloneFeature,
]

AnyInput = Union[
    DiscreteInput,
    CategoricalInput,
    ContinuousInput,
    ContinuousDescriptorInput,
    CategoricalDescriptorInput,
    CategoricalMolecularInput,
    TaskInput,
    ContinuousMolecularInput,
]

AnyOutput = Union[ContinuousOutput, CategoricalOutput]

_ENGINEERED_FEATURE_TYPES: List[Type[EngineeredFeature]] = [
    SumFeature,
    MeanFeature,
    WeightedMeanFeature,
    WeightedSumFeature,
    MolecularWeightedMeanFeature,
    MolecularWeightedSumFeature,
    ProductFeature,
    InterpolateFeature,
    CloneFeature,
]

AnyEngineeredFeature = Union[tuple(_ENGINEERED_FEATURE_TYPES)]


def register_engineered_feature(data_model_cls: Type[EngineeredFeature]) -> None:
    """Register a custom engineered feature type so it is accepted in EngineeredFeatures.

    This appends the type to the internal registry, rebuilds the
    ``AnyEngineeredFeature`` union, and calls ``model_rebuild`` on
    ``EngineeredFeatures`` so that Pydantic accepts the new type.

    Args:
        data_model_cls: A concrete subclass of ``EngineeredFeature``.
    """
    global AnyEngineeredFeature
    if data_model_cls in _ENGINEERED_FEATURE_TYPES:
        return
    _ENGINEERED_FEATURE_TYPES.append(data_model_cls)
    AnyEngineeredFeature = Union[tuple(_ENGINEERED_FEATURE_TYPES)]

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
