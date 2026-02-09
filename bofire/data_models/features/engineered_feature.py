from abc import abstractmethod
from typing import TYPE_CHECKING, Annotated, ClassVar, List, Literal

from pydantic import Field, PositiveFloat, PositiveInt, model_validator

from bofire.data_models.features.api import ContinuousDescriptorInput, ContinuousInput
from bofire.data_models.features.feature import Feature
from bofire.data_models.features.molecular import ContinuousMolecularInput
from bofire.data_models.molfeatures.api import AnyMolFeatures
from bofire.data_models.types import Bounds, Descriptors, FeatureKeys


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
        return len(self.descriptors)

    def validate_features(self, inputs: "Inputs"):
        super().validate_features(inputs)
        for feature_key in self.features:
            feature = inputs.get_by_key(feature_key)
            if not isinstance(feature, ContinuousDescriptorInput):
                raise ValueError(
                    f"Feature '{feature_key}' is not a ContinuousDescriptorInput",
                )
            if len(set(self.descriptors) - set(feature.descriptors)) > 0:
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
    molfeatures: AnyMolFeatures
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


class ProductFeature(EngineeredFeature):
    """Product feature, which compute the sum over the specified features.

    Args:
        features: The features to be used to compute the product.
            It is allowed to state a feature more than once to for example
            an quadratic term.
        keep_features: Whether to keep the original features after
            creating the engineered feature in surrogate creation.
    """

    type: Literal["ProductFeature"] = "ProductFeature"
    order_id: ClassVar[int] = 4
    features: Annotated[List[str], Field(min_length=2)]

    @property
    def n_transformed_inputs(self) -> int:
        return 1


class InterpolateFeature(EngineeredFeature):
    """Interpolation feature, which performs piecewise linear interpolation
    over specified x and y coordinate features.

    Args:
        x_keys: Feature keys used as x-coordinates for interpolation.
        y_keys: Feature keys used as y-coordinates for interpolation.
        interpolation_range: (lower, upper) bounds for the interpolation x-grid.
        n_interpolation_points: Number of evenly spaced points in the interpolation grid.
        prepend_x: Extra x-values to prepend before the feature x-values.
        append_x: Extra x-values to append after the feature x-values.
        prepend_y: Extra y-values to prepend before the feature y-values.
        append_y: Extra y-values to append after the feature y-values.
        normalize_y: Divisor for y-values before interpolation.
        normalize_x: Whether to normalize x-values to [0, 1] before interpolation.
    """

    type: Literal["InterpolateFeature"] = "InterpolateFeature"
    order_id: ClassVar[int] = 5

    x_keys: List[str]
    y_keys: List[str]
    interpolation_range: Bounds
    n_interpolation_points: PositiveInt

    prepend_x: List[float] = Field(default_factory=list)
    append_x: List[float] = Field(default_factory=list)
    prepend_y: List[float] = Field(default_factory=list)
    append_y: List[float] = Field(default_factory=list)
    normalize_y: PositiveFloat = 1.0
    normalize_x: bool = False

    @model_validator(mode="after")
    def validate_keys(self) -> "InterpolateFeature":
        if set(self.x_keys) & set(self.y_keys):
            raise ValueError("x_keys and y_keys must not overlap.")
        if sorted(self.features) != sorted(self.x_keys + self.y_keys):
            raise ValueError("features must match x_keys + y_keys.")
        n_x = len(self.x_keys) + len(self.prepend_x) + len(self.append_x)
        n_y = len(self.y_keys) + len(self.prepend_y) + len(self.append_y)
        if n_x != n_y:
            raise ValueError("Total number of x and y values must be equal.")
        return self

    @property
    def n_transformed_inputs(self) -> int:
        return self.n_interpolation_points
