from bofire.data_models.features._register import register_engineered_feature
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
from bofire.data_models.unions import tagged_union


AnyFeature = tagged_union(
    DiscreteInput,
    CategoricalInput,
    ContinuousInput,
    ContinuousOutput,
    CategoricalOutput,
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
)

AnyInput = tagged_union(
    DiscreteInput,
    CategoricalInput,
    ContinuousInput,
    ContinuousDescriptorInput,
    CategoricalDescriptorInput,
    CategoricalMolecularInput,
    TaskInput,
    ContinuousMolecularInput,
)

AnyOutput = tagged_union(ContinuousOutput, CategoricalOutput)

_ENGINEERED_FEATURE_TYPES: list[type[EngineeredFeature]] = [
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

AnyEngineeredFeature = tagged_union(*_ENGINEERED_FEATURE_TYPES)
