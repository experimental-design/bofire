from abc import abstractmethod
from typing import TYPE_CHECKING, ClassVar, Literal

from bofire.data_models.features.api import ContinuousDescriptorInput, ContinuousInput
from bofire.data_models.features.feature import Feature
from bofire.data_models.features.molecular import ContinuousMolecularInput
from bofire.data_models.molfeatures.api import MordredDescriptors
from bofire.data_models.types import Descriptors, FeatureKeys


if TYPE_CHECKING:
    from bofire.data_models.domain.api import Inputs


class EngineeredFeature(Feature):
    """Base class for an engineered feature.

    Args:
        features: The features to be used to compute the engineered feature.
        keep_features: Whether to keep the original features after
            creating the engineered feature in surrogate creation.
    """

    features: FeatureKeys
    keep_features: bool = True

    def validate_features(self, inputs: "Inputs"):
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

    def _validate_features(self, inputs: "Inputs"):
        pass

    @property
    @abstractmethod
    def n_transformed_inputs(self) -> int:
        pass


class SumFeature(EngineeredFeature):
    """Sum feature, which computes the sum over the specified features.

    Args:
        features: The features to be used to compute the sum.
        keep_features: Whether to keep the original features after
            creating the engineered feature in surrogate creation.
    """

    type: Literal["SumFeature"] = "SumFeature"
    order_id: ClassVar[int] = 0

    @property
    def n_transformed_inputs(self) -> int:
        return 1


class MeanFeature(EngineeredFeature):
    """Mean feature, which computes the mean over the specified features.

    Args:
        features: The features to be used to compute the mean.
        keep_features: Whether to keep the original features after
            creating the engineered feature in surrogate creation.
    """

    type: Literal["MeanFeature"] = "MeanFeature"
    order_id: ClassVar[int] = 1

    @property
    def n_transformed_inputs(self) -> int:
        return 1


class WeightedSumFeature(EngineeredFeature):
    """Weighted sum feature, which computes the sum over the specified
    descriptors weighted by the involved feature values.

    Args:
        features: The features to be used to compute the weighted sum.
        descriptors: The descriptors to be used to compute the weighted sum.
        keep_features: Whether to keep the original features after
            creating the engineered feature in surrogate creation.
    """

    type: Literal["WeightedSumFeature"] = "WeightedSumFeature"
    descriptors: Descriptors
    order_id: ClassVar[int] = 2

    @property
    def n_transformed_inputs(self) -> int:
        return len(self.descriptors)  # type: ignore

    def validate_features(self, inputs: "Inputs"):
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


class MolecularWeightedSumFeature(EngineeredFeature):
    """Molecular weighted sum feature, which computes the sum over the specified
    molecular descriptors weighted by the involved feature values.

    Args:
        features: The molecular features to be used to compute the weighted sum.
        molfeatures: The molecular feature descriptor specification.
        keep_features: Whether to keep the original features after
            creating the engineered feature in surrogate creation.
    """

    type: Literal["MolecularWeightedSumFeature"] = "MolecularWeightedSumFeature"
    molfeatures: MordredDescriptors
    order_id: ClassVar[int] = 3

    @property
    def n_transformed_inputs(self) -> int:
        return len(self.molfeatures.get_descriptor_names())

    def validate_features(self, inputs: "Inputs"):
        super().validate_features(inputs)
        for feature_key in self.features:
            feature = inputs.get_by_key(feature_key)
            if not isinstance(feature, ContinuousMolecularInput):
                raise ValueError(
                    f"Feature '{feature_key}' is not a ContinuousMolecularInput",
                )
