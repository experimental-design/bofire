from abc import abstractmethod
from typing import TYPE_CHECKING, Annotated, ClassVar, List, Literal

import pandas as pd
from pydantic import Field, PositiveFloat, PositiveInt, model_validator

from bofire.data_models.encodings.naming import get_encoded_name
from bofire.data_models.features._descriptor_spec import DescriptorSpec
from bofire.data_models.features.api import ContinuousInput
from bofire.data_models.features.descriptors import Descriptors
from bofire.data_models.features.feature import Feature
from bofire.data_models.types import Bounds, FeatureKeys, OneFeatureKeys


if TYPE_CHECKING:
    from bofire.data_models.domain.api import Inputs


class EngineeredFeature(Feature):
    """Base class for an engineered feature.

    An engineered feature is not proposed or measured. It is computed from other input
    features and appended as extra columns when the surrogate is built, which lets a
    model see a derived quantity without it becoming a degree of freedom.
    """

    features: FeatureKeys = Field(
        description="Keys of the input features this feature is computed from.",
    )
    keep_features: bool = Field(
        default=True,
        description="Whether the source features are also passed to the surrogate. "
        "Set to false to replace them with the engineered feature rather than "
        "supplying both.",
    )

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

        Resolved against ``inputs`` on every call. ``len(get_names(inputs))`` is the
        feature's width, and is what offset bookkeeping uses.
        """


class SumFeature(EngineeredFeature):
    """Sum feature, which computes the sum over the specified features.

    Appends one column holding the total, for example the overall amount of a set of
    ingredients.

    Examples:
        >>> SumFeature(key="total_solvent", features=["water", "ethanol"])
    """

    type: Literal["SumFeature"] = "SumFeature"
    order_id: ClassVar[int] = 0

    def get_names(self, inputs: "Inputs") -> List[str]:
        return [self.key]


class MeanFeature(EngineeredFeature):
    """Mean feature, which computes the mean over the specified features.

    Appends one column holding the average of the source features.

    Examples:
        >>> MeanFeature(key="mean_temperature", features=["T_start", "T_end"])
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

    This is how a mixture gets described by the properties of what is in it rather than
    by the amounts alone: each component contributes its descriptor row in proportion to
    how much of it is present.

    The component features named here must each carry a one-row ``Descriptors`` block,
    as ``ContinuousInput`` does in its second example.

    Examples:
        Blend two stored descriptor columns of the components:

        >>> WeightedSumFeature(
        ...     key="solvent_properties",
        ...     features=["water", "ethanol"],
        ...     columns=["logP", "MW"],
        ... )

        Blend generated columns instead, running a generator over the components'
        SMILES structures — ``columns=[]`` selects none of the stored ones:

        >>> WeightedSumFeature(
        ...     key="solvent_fingerprint",
        ...     features=["water", "ethanol"],
        ...     columns=[],
        ...     generators=[Fingerprints(n_bits=32)],
        ...     normalize=True,
        ... )
    """

    type: Literal["WeightedSumFeature"] = "WeightedSumFeature"
    order_id: ClassVar[int] = 2
    normalize: bool = Field(
        default=False,
        description="Whether to divide by the total amount, giving a weighted mean "
        "instead of a weighted sum. Use this when the composition matters but the "
        "overall scale does not.",
    )

    def component_table(self, features: List["ContinuousInput"]) -> pd.DataFrame:
        """Descriptor table with one row per component feature (weighted-sum scope).

        The components' one-row blocks are stacked into a single block, so generators run
        once over the combined structures and every row shares the same columns.
        """
        return self.build(self._merged_descriptors(features), self._index(features))

    @staticmethod
    def _merged_descriptors(features: List["ContinuousInput"]) -> Descriptors:
        """The components' one-row blocks as a single block, one row per component.

        ``Descriptors.concat`` rejects components that carry no block, or that disagree
        on their columns or on carrying a structure.
        """
        return Descriptors.concat([f.descriptors for f in features])

    @staticmethod
    def _index(features: List["ContinuousInput"]) -> List[str]:
        """Row labels of the blended block: the component keys."""
        return [f.key for f in features]

    def get_names(self, inputs: "Inputs") -> List[str]:
        """One name per descriptor column of the blended block."""
        components = [inputs.get_by_key(key) for key in self.features]
        names = self.resolved_names(
            self._merged_descriptors(components),
            self._index(components),
        )
        return [get_encoded_name(self.key, name) for name in names]

    def validate_features(self, inputs: "Inputs"):
        """Gate the spec against each component *and* against the blend they merge into.

        Per component first, so an incompatible one is named in the message. Then against
        the blended block, which is what the consumers actually read: merging is where the
        components' mutual compatibility — same columns, all or none carrying a structure —
        is decided, and without this it would only surface at build time.
        """
        super().validate_features(inputs)
        components = [inputs.get_by_key(key) for key in self.features]
        for feat in components:
            self.validate_for(feat.descriptors, feat.key)
        self.validate_for(self._merged_descriptors(components), self.key)


class ProductFeature(EngineeredFeature):
    """Product feature, which computes the product over the specified features.

    Appends one column holding the product, which is how an interaction between inputs
    is made available to a model that would otherwise only see them separately.

    Examples:
        >>> ProductFeature(key="temp_x_time", features=["temperature", "time"])

        Repeat a key for a power term:

        >>> ProductFeature(key="temp_squared", features=["temperature", "temperature"])
    """

    type: Literal["ProductFeature"] = "ProductFeature"
    order_id: ClassVar[int] = 4
    features: Annotated[List[str], Field(min_length=2)] = Field(
        description="Keys of the input features to multiply. A key may be repeated to "
        "raise that feature to a power, so ['x', 'x'] gives a quadratic term.",
    )

    def get_names(self, inputs: "Inputs") -> List[str]:
        return [self.key]


class InterpolateFeature(EngineeredFeature):
    """Interpolation feature, which performs piecewise linear interpolation
    over specified x and y coordinate features.

    The optimized inputs are the coordinates of a curve; this feature interpolates that
    curve on a fixed grid and appends one column per grid point. That gives the model a
    consistent representation of the curve even though the coordinates move, which is
    how profiles such as a temperature ramp can be optimized.

    Examples:
        A temperature ramp given by two moving (time, temperature) points, evaluated on
        a five-point grid:

        >>> InterpolateFeature(
        ...     key="ramp",
        ...     features=["t1", "t2", "T1", "T2"],
        ...     x_keys=["t1", "t2"],
        ...     y_keys=["T1", "T2"],
        ...     interpolation_range=[0.0, 1.0],
        ...     n_interpolation_points=5,
        ... )
    """

    type: Literal["InterpolateFeature"] = "InterpolateFeature"
    order_id: ClassVar[int] = 5

    x_keys: List[str] = Field(
        description="Keys of the input features holding the x-coordinates of the "
        "curve. Must not overlap `y_keys`, and together with them must cover exactly "
        "`features`.",
    )
    y_keys: List[str] = Field(
        description="Keys of the input features holding the y-coordinates of the "
        "curve, paired positionally with the x-coordinates.",
    )
    interpolation_range: Bounds = Field(
        description="Lower and upper end of the grid the curve is evaluated on. Must "
        "be [0, 1] when `normalize_x` is enabled.",
    )
    n_interpolation_points: PositiveInt = Field(
        description="Number of evenly spaced grid points, and therefore the number of "
        "columns this feature appends.",
    )

    prepend_x: List[float] = Field(
        default=[],
        description="Fixed x-coordinates placed before those taken from `x_keys`, for "
        "anchoring the curve at a known starting point that is not optimized.",
    )
    append_x: List[float] = Field(
        default=[],
        description="Fixed x-coordinates placed after those taken from `x_keys`.",
    )
    prepend_y: List[float] = Field(
        default=[],
        description="Fixed y-coordinates placed before those taken from `y_keys`. The "
        "total number of x- and y-coordinates must match.",
    )
    append_y: List[float] = Field(
        default=[],
        description="Fixed y-coordinates placed after those taken from `y_keys`.",
    )
    normalize_y: PositiveFloat = Field(
        default=1.0,
        description="Divisor applied to the y-coordinates before interpolation, for "
        "bringing the curve onto a comparable scale.",
    )
    normalize_x: bool = Field(
        default=False,
        description="Whether to rescale the x-coordinates onto [0, 1] before "
        "interpolating, which makes the curve's shape independent of its extent. "
        "Requires `interpolation_range` to be [0, 1].",
    )

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

    Examples:
        >>> CloneFeature(key="temperature_copy", features=["temperature"])
    """

    type: Literal["CloneFeature"] = "CloneFeature"
    order_id: ClassVar[int] = 5
    features: OneFeatureKeys = Field(
        description="Keys of the input features to copy. Each appears once; the copy "
        "gets its own column so it can be scaled or fed to a different kernel than the "
        "original.",
    )

    def get_names(self, inputs: "Inputs") -> List[str]:
        return [get_encoded_name(self.key, key) for key in self.features]
