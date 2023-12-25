from typing import Union

from bofire.data_models.features.categorical import CategoricalInput, CategoricalOutput
from bofire.data_models.features.continuous import ContinuousInput, ContinuousOutput
from bofire.data_models.features.descriptor import (
    CategoricalDescriptorInput,
    ContinuousDescriptorInput,
)
from bofire.data_models.features.discrete import DiscreteInput
from bofire.data_models.features.feature import (
    _CAT_SEP,
    Feature,
    Input,
    Output,
    TInputTransformSpecs,
)
from bofire.data_models.features.molecular import (
    CategoricalMolecularInput,
    MolecularInput,
)
from bofire.data_models.features.numerical import NumericalInput

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
    MolecularInput,
    CategoricalMolecularInput,
]

AnyInput = Union[
    DiscreteInput,
    CategoricalInput,
    ContinuousInput,
    ContinuousDescriptorInput,
    CategoricalDescriptorInput,
    MolecularInput,
    CategoricalMolecularInput,
]

AnyOutput = Union[ContinuousOutput, CategoricalOutput]
