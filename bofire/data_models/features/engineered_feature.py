from abc import abstractmethod
from typing import TYPE_CHECKING, Annotated, ClassVar, List, Literal

import pandas as pd
from pydantic import Field, PositiveFloat, PositiveInt, model_validator

from bofire.data_models.encodings.naming import get_encoded_name
from bofire.data_models.features._descriptor_spec import DescriptorSpec
from bofire.data_models.features._descriptors import Descriptors
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

    @abstractmethod
    def get_names(self, inputs: "Inputs") -> List[str]:
        """Names of the columns this feature appends, given the input features.

        Resolved on demand from ``inputs`` rather than stored, mirroring
        :meth:`CategoricalEncoding.get_names`. The width used for offset bookkeeping is
        ``len(get_names(inputs))``, so it is always derived from the same data the mapper
        builds its columns from and cannot go stale.
        """


class SumFeature(EngineeredFeature):
    """Sum feature, which computes the sum over the specified features.

    Args:
        features: The features to be used to compute the sum.
        keep_features: Whether to keep the original features after
            creating the engineered feature in surrogate creation.
    """

    type: Literal["SumFeature"] = "SumFeature"
    order_id: ClassVar[int] = 0

    def get_names(self, inputs: "Inputs") -> List[str]:
        return [self.key]


class MeanFeature(EngineeredFeature):
    """Mean feature, which computes the mean over the specified features.

    Args:
        features: The features to be used to compute the mean.
        keep_features: Whether to keep the original features after
            creating the engineered feature in surrogate creation.
    """

    type: Literal["MeanFeature"] = "MeanFeature"
    order_id: ClassVar[int] = 1

    def get_names(self, inputs: "Inputs") -> List[str]:
        return [self.key]


class WeightedSumFeature(EngineeredFeature, DescriptorSpec):
    """Amount-weighted blend of descriptors over the specified component features.

    For each descriptor ``d`` the output is ``Σᵢ amountᵢ · rowᵢ,d`` where ``amountᵢ``
    is the value of component feature ``i`` (optionally normalized by ``Σᵢ amountᵢ``).
    The descriptor columns are declared by the :class:`DescriptorSpec` mixin
    (``columns`` for static columns and/or ``generators`` for molecular generators).

    Args:
        features: The component features to blend.
        columns / generators / filter_descriptors: see :class:`DescriptorSpec`.
        normalize: If True, divide by the sum of amounts (weighted mean).
        keep_features: Whether to keep the original features in surrogate creation.
    """

    type: Literal["WeightedSumFeature"] = "WeightedSumFeature"
    order_id: ClassVar[int] = 2
    normalize: bool = False

    def component_table(self, features: List["ContinuousInput"]) -> pd.DataFrame:
        """Descriptor table with one row per component feature (weighted-sum scope).

        The components' one-row blocks are stacked into a single block, so generators run
        once over the combined structures and every row shares the same columns.
        """
        return self.build(self._block(features), [f.key for f in features])

    @staticmethod
    def _block(features: List["ContinuousInput"]) -> Descriptors:
        """The components' blocks as one block (Descriptors.concat rejects missing ones)."""
        return Descriptors.concat([f.descriptors for f in features])

    def get_names(self, inputs: "Inputs") -> List[str]:
        """One name per descriptor column of the blended block.

        Only correlation filtering needs the assembled *values* — and therefore the
        generators (rdkit). Without it the columns are exactly the static and generator
        names ``build`` concatenates, so they come from metadata alone.
        """
        components = [inputs.get_by_key(key) for key in self.features]
        names = (
            list(self.component_table(components).columns)
            if self.filter_descriptors
            else self.column_names(self._block(components))
        )
        return [get_encoded_name(self.key, name) for name in names]

    def validate_features(self, inputs: "Inputs"):
        super().validate_features(inputs)
        for key in self.features:
            feat = inputs.get_by_key(key)
            self.validate_for(feat.descriptors, feat.key)


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

    def get_names(self, inputs: "Inputs") -> List[str]:
        return [self.key]


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

    def get_names(self, inputs: "Inputs") -> List[str]:
        return [
            get_encoded_name(self.key, str(i))
            for i in range(self.n_interpolation_points)
        ]


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

    def get_names(self, inputs: "Inputs") -> List[str]:
        return [get_encoded_name(self.key, key) for key in self.features]
