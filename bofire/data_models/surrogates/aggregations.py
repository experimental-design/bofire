from abc import abstractmethod
from typing import Annotated, Any, List, Literal, Union

from pydantic import Field

from bofire.data_models.base import BaseModel
from bofire.data_models.domain.api import Inputs
from bofire.data_models.features.api import ContinuousDescriptorInput, ContinuousInput
from bofire.data_models.types import FeatureKeys


class Aggregation(BaseModel):
    type: Any
    key: str
    features: FeatureKeys
    keep_features: bool = False

    def validate_features(self, inputs: Inputs):
        missing_features = [
            feature
            for feature in self.features
            if feature not in inputs.get_keys(ContinuousInput)
        ]
        if missing_features:
            raise ValueError(
                f"The following features are missing in inputs: {missing_features}"
            )
        self._validate_features(inputs)

    @abstractmethod
    def _validate_features(self, inputs: Inputs):
        pass

    @property
    @abstractmethod
    def n_outputs(self) -> int:
        pass


class OnlyContinuousMixin:
    def _validate_features(self, inputs: Inputs):
        pass


class SumAggregation(Aggregation):
    type: Literal["SumAggregation"] = "SumAggregation"

    @property
    def n_outputs(self) -> int:
        return 1


class MeanAggregation(Aggregation):
    type: Literal["MeanAggregation"] = "MeanAggregation"

    @property
    def n_outputs(self) -> int:
        return 1


class OnlyDescriptorsMixin:
    def _validate_features(self, inputs: Inputs):
        for feature_key in self.features:
            feature = inputs.get_by_key(feature_key)
            if not isinstance(feature, ContinuousDescriptorInput):
                raise ValueError(
                    f"Feature '{feature_key}' is not a ContinuousDescriptorInput",
                )
            if len(set(self.descriptors) - set(feature.descriptors)) > 0:  # type: ignore
                raise ValueError(
                    f"Not all descriptors {self.descriptors} are present in feature '{feature_key}'",
                )


class WeightedSumAggregation(Aggregation):
    type: Literal["WeightedSumAggregation"] = "WeightedSumAggregation"
    descriptors: Annotated[List[float], Field(min_length=2)]

    @property
    def n_outputs(self) -> int:
        return len(self.descriptors)


AnyAggregation = Union[SumAggregation, MeanAggregation]
