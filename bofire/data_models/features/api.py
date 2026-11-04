from bofire.data_models.features._register import register_engineered_feature
from bofire.data_models.features.categorical import CategoricalInput, CategoricalOutput
from bofire.data_models.features.continuous import ContinuousInput, ContinuousOutput
from bofire.data_models.features.descriptors import Descriptors
from bofire.data_models.features.discrete import DiscreteInput
from bofire.data_models.features.engineered_feature import (
    CloneFeature,
    EngineeredFeature,
    InterpolateFeature,
    MeanFeature,
    ProductFeature,
    SumFeature,
    WeightedSumFeature,
)
from bofire.data_models.features.feature import Feature, Input, Output
from bofire.data_models.features.numerical import NumericalInput
from bofire.data_models.features.task import (
    CategoricalTaskInput,
    ContinuousTaskInput,
    TaskInput,
)
from bofire.data_models.unions import tagged_union


_FEATURE_TYPES: list[type[Feature]] = [
    DiscreteInput,
    CategoricalInput,
    ContinuousInput,
    ContinuousOutput,
    CategoricalOutput,
    CategoricalTaskInput,
    ContinuousTaskInput,
    SumFeature,
    MeanFeature,
    WeightedSumFeature,
    ProductFeature,
    InterpolateFeature,
    CloneFeature,
]

AnyFeature = tagged_union(*_FEATURE_TYPES)

AnyInput = tagged_union(
    DiscreteInput,
    CategoricalInput,
    ContinuousInput,
    ContinuousTaskInput,
    CategoricalTaskInput,
)

AnyOutput = tagged_union(ContinuousOutput, CategoricalOutput)

_ENGINEERED_FEATURE_TYPES: list[type[EngineeredFeature]] = [
    SumFeature,
    MeanFeature,
    WeightedSumFeature,
    ProductFeature,
    InterpolateFeature,
    CloneFeature,
]

AnyEngineeredFeature = tagged_union(*_ENGINEERED_FEATURE_TYPES)
