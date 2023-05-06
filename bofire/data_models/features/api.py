from typing import Union

from bofire.data_models.features.categorical import CategoricalInput, CategoricalOutput
from bofire.data_models.features.continuous import ContinuousInput, ContinuousOutput
from bofire.data_models.features.descriptor import (
    CategoricalDescriptorInput,
    ContinuousDescriptorInput,
)
from bofire.data_models.features.discrete import DiscreteInput
from bofire.data_models.features.feature import (  # noqa: F401
    _CAT_SEP,
    Feature,
    Input,
    Output,
    TInputTransformSpecs,
)
from bofire.data_models.features.molecular import MolecularInput
from bofire.data_models.features.numerical import NumericalInput

AbstractFeature = Union[
    Feature,
    Input,
    Output,
    NumericalInput,
]

AnyFeature = Union[
    DiscreteInput,
    CategoricalInput,
    ContinuousOutput,
    ContinuousInput,
    ContinuousDescriptorInput,
    CategoricalDescriptorInput,
    MolecularInput,
    CategoricalOutput,
]

AnyInput = Union[
    DiscreteInput,
    CategoricalInput,
    ContinuousInput,
    ContinuousDescriptorInput,
    CategoricalDescriptorInput,
    MolecularInput,
]

AnyOutput = Union[ContinuousOutput, CategoricalOutput]
