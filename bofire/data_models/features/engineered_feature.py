import warnings
from abc import abstractmethod
from typing import TYPE_CHECKING, Annotated, ClassVar, List, Literal

from pydantic import Field, PositiveFloat, PositiveInt, model_validator

from bofire.data_models.descriptors.api import AnyDescriptorSource
from bofire.data_models.features.api import ContinuousInput
from bofire.data_models.features.feature import Feature
from bofire.data_models.types import Bounds, FeatureKeys, OneFeatureKeys


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
    """Amount-weighted blend of descriptors over the specified component features.

    For each descriptor ``d`` the output is ``Σᵢ amountᵢ · rowᵢ,d`` where ``amountᵢ``
    is the value of component feature ``i`` (optionally normalized by ``Σᵢ amountᵢ``).
    The ``source`` decides how each component's descriptor row is produced (static
    columns / molecular generator / composite).

    Args:
        features: The component features to blend.
        source: The descriptor source (read from each component's descriptors).
        normalize: If True, divide by the sum of amounts (weighted mean).
        keep_features: Whether to keep the original features in surrogate creation.
    """

    type: Literal["WeightedSumFeature"] = "WeightedSumFeature"
    order_id: ClassVar[int] = 2
    source: AnyDescriptorSource
    normalize: bool = False

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_descriptors(cls, data):
        # legacy shape: descriptors=[names] (static) and no source
        if isinstance(data, dict) and "source" not in data and "descriptors" in data:
            warnings.warn(
                "`descriptors=` on WeightedSumFeature is deprecated; use "
                "`source=StaticSource(columns=...)` instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            data["source"] = {
                "type": "StaticSource",
                "columns": data.pop("descriptors"),
            }
        return data

    @property
    def n_transformed_inputs(self) -> int:
        declared = self.source.declared_names()
        return len(declared) if declared is not None else 0

    def validate_features(self, inputs: "Inputs"):
        super().validate_features(inputs)
        for feature_key in self.features:
            self.source.check(inputs.get_by_key(feature_key))


class WeightedMeanFeature(WeightedSumFeature):
    """Deprecated. Use :class:`WeightedSumFeature` with ``normalize=True``."""

    type: Literal["WeightedMeanFeature"] = "WeightedMeanFeature"
    order_id: ClassVar[int] = 6

    @model_validator(mode="before")
    @classmethod
    def _migrate_mean(cls, data):
        if isinstance(data, dict):
            warnings.warn(
                "`WeightedMeanFeature` is deprecated, use "
                "`WeightedSumFeature(normalize=True)` instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            data.setdefault("normalize", True)
        return data


class MolecularWeightedSumFeature(WeightedSumFeature):
    """Deprecated. Use :class:`WeightedSumFeature` with a ``GeneratedSource``."""

    type: Literal["MolecularWeightedSumFeature"] = "MolecularWeightedSumFeature"
    order_id: ClassVar[int] = 3

    @model_validator(mode="before")
    @classmethod
    def _migrate_molecular(cls, data):
        if isinstance(data, dict) and "source" not in data and "molfeatures" in data:
            warnings.warn(
                "`MolecularWeightedSumFeature` is deprecated, use "
                "`WeightedSumFeature(source=GeneratedSource(generator=...))`.",
                DeprecationWarning,
                stacklevel=2,
            )
            data["source"] = {
                "type": "GeneratedSource",
                "structure": "smiles",
                "generator": data.pop("molfeatures"),
            }
        return data


class MolecularWeightedMeanFeature(MolecularWeightedSumFeature):
    """Deprecated. Use :class:`WeightedSumFeature` with a ``GeneratedSource`` and
    ``normalize=True``."""

    type: Literal["MolecularWeightedMeanFeature"] = "MolecularWeightedMeanFeature"
    order_id: ClassVar[int] = 7

    @model_validator(mode="before")
    @classmethod
    def _migrate_mean(cls, data):
        if isinstance(data, dict):
            data.setdefault("normalize", True)
        return data


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
        if self.normalize_x and tuple(self.interpolation_range) != (0.0, 1.0):
            raise ValueError(
                "When normalize_x is True, interpolation_range must be (0, 1) "
                "since x-values are normalized to [0, 1]."
            )
        return self

    @property
    def n_transformed_inputs(self) -> int:
        return self.n_interpolation_points


class CloneFeature(EngineeredFeature):
    """Engineered feature that creates a copy of the original features.

    This is useful if you want to have features undergoing different scalers
    before entering different kernels.

    Args:
        features: The features to be used to compute the product.
            It is allowed to state a feature more than once to for example
            an quadratic term.
        keep_features: Whether to keep the original features after
            creating the engineered feature in surrogate creation.
    """

    type: Literal["CloneFeature"] = "CloneFeature"
    order_id: ClassVar[int] = 5
    features: OneFeatureKeys

    @property
    def n_transformed_inputs(self) -> int:
        return len(self.features)
