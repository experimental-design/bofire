from typing import List, Tuple, Union

from bofire.domain import features

AnyFeature = Union[
    features.DiscreteInput,
    features.ContinuousInput,
    features.ContinuousDescriptorInput,
    features.CategoricalInput,
    features.CategoricalDescriptorInput,
    features.ContinuousOutput,
]

AnyInputFeature = Union[
    features.ContinuousInput,
    features.DiscreteInput,
    features.ContinuousDescriptorInput,
    features.CategoricalInput,
    features.CategoricalDescriptorInput,
]

AnyOutputFeature = features.ContinuousOutput

FeatureSequence = Union[List[AnyFeature], Tuple[AnyFeature]]
