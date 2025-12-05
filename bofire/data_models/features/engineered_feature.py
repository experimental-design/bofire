from abc import abstractmethod
from typing import Annotated, ClassVar, List, Literal

from pydantic import Field

from bofire.data_models.features.api import ContinuousDescriptorInput, ContinuousInput
from bofire.data_models.features.feature import Feature
from bofire.data_models.types import FeatureKeys


class EngineeredFeature(Feature):
    features: FeatureKeys
    keep_features: bool = True

    def validate_features(self, inputs: "Inputs"):  # noqa: F821
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

    def _validate_features(self, inputs: "Inputs"):  # noqa: F821
        pass

    @property
    @abstractmethod
    def n_outputs(self) -> int:
        pass


class SumFeature(EngineeredFeature):
    type: Literal["SumFeature"] = "SumFeature"
    order_id: ClassVar[int] = 0

    @property
    def n_outputs(self) -> int:
        return 1


class MeanFeature(EngineeredFeature):
    type: Literal["MeanFeature"] = "MeanFeature"
    order_id: ClassVar[int] = 1

    @property
    def n_outputs(self) -> int:
        return 1


class WeightedSumFeature(EngineeredFeature):
    type: Literal["WeightedSumFeature"] = "WeightedSumFeature"
    descriptors: Annotated[List[str], Field(min_length=2)]
    order_id: ClassVar[int] = 2

    @property
    def n_outputs(self) -> int:
        return len(self.descriptors)

    def validate_features(self, inputs: "Inputs"):  # noqa: F821
        super().validate_features(inputs)
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
