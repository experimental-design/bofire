from typing import List, Tuple, Union

from bofire.domain import feature

AnyFeature = Union[
    feature.DiscreteInput,
    feature.ContinuousInput,
    feature.ContinuousDescriptorInput,
    feature.CategoricalInput,
    feature.CategoricalDescriptorInput,
    feature.ContinuousOutput,
]

AnyInputFeature = Union[
    feature.ContinuousInput,
    feature.DiscreteInput,
    feature.ContinuousDescriptorInput,
    feature.CategoricalInput,
    feature.CategoricalDescriptorInput,
]

AnyOutputFeature = feature.ContinuousOutput

FeatureSequence = Union[List[AnyFeature], Tuple[AnyFeature]]
