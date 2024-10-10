from typing import Union

from bofire.data_models.features.categorical import CategoricalInput, CategoricalOutput
from bofire.data_models.features.continuous import ContinuousInput, ContinuousOutput
from bofire.data_models.features.descriptor import (
    CategoricalDescriptorInput,
    ContinuousDescriptorInput,
)
from bofire.data_models.features.discrete import DiscreteInput
from bofire.data_models.features.feature import Feature, Input, Output
from bofire.data_models.features.molecular import (
    CategoricalMolecularInput,
    MolecularInput,
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
    MolecularInput,
    CategoricalMolecularInput,
    TaskInput,
]

AnyInput = Union[
    DiscreteInput,
    CategoricalInput,
    ContinuousInput,
    ContinuousDescriptorInput,
    CategoricalDescriptorInput,
    MolecularInput,
    CategoricalMolecularInput,
    TaskInput,
]

AnyOutput = Union[ContinuousOutput, CategoricalOutput]
