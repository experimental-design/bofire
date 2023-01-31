from typing import List, Tuple, Union

from bofire.domain.features import (
    CategoricalDescriptorInput,
    CategoricalInput,
    ContinuousDescriptorInput,
    ContinuousInput,
    ContinuousOutput,
    DiscreteInput,
)

AnyFeature = Union[
    # InputFeature,
    # NumericalInputFeature,
    DiscreteInput,
    ContinuousInput,
    ContinuousDescriptorInput,
    CategoricalInput,
    CategoricalDescriptorInput,
    # OutputFeature,
    ContinuousOutput,
]
AnyInputFeature = Union[
    # InputFeature,
    # NumericalInputFeature,
    ContinuousInput,
    DiscreteInput,
    ContinuousDescriptorInput,
    CategoricalInput,
    CategoricalDescriptorInput,
]
AnyOutputFeature = ContinuousOutput
# Union[OutputFeature, ContinuousOutput,]

FeatureSequence = Union[List[AnyFeature], Tuple[AnyFeature]]
