import operator as ops
from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Callable, List, Literal, Union

import pandas as pd
from numpy.typing import ArrayLike

from bofire.data_models.base import BaseModel
from bofire.data_models.features.categorical import CategoricalInput
from bofire.data_models.features.continuous import ContinuousInput
from bofire.data_models.features.discrete import DiscreteInput
from bofire.data_models.features.molecular import MolecularInput


if TYPE_CHECKING:
    from bofire.data_models.features.api import AnyInput


threshold_operators: dict[str, Callable] = {
    "<": ops.lt,
    "<=": ops.le,
    ">": ops.gt,
    ">=": ops.ge,
}


class Condition(BaseModel):
    """Base class for all conditions.

    Conditions evaluate an expression regarding a single feature.
    Conditions are part of the CategoricalExcludeConstraint.
    """

    type: Any

    @abstractmethod
    def __call__(self, values: pd.Series) -> pd.Series:
        """Evaluate the condition on a given data series.

        Args:
            values: A series containing feature values.

        Returns:
            A Boolean series indicating which elements satisfy the condition.
        """

    @abstractmethod
    def validate_feature(self, feature: "AnyInput") -> None:
        """Validate that a feature is compatible with this condition."""


class ThresholdCondition(Condition):
    """Class for modelling threshold conditions.

    It can only be applied to ContinuousInput and DiscreteInput features. It is
    checked if the feature value is above or below a certain threshold depending on the
    operator. If the expression evaluated to true, the condition is fulfilled.

    Attributes:
        threshold: Threshold value.
        operator: Operator to use for comparison. Can be one of "<", "<=", ">", ">=".
    """

    type: Literal["ThresholdCondition"] = "ThresholdCondition"
    threshold: float
    operator: Literal["<", "<=", ">", ">="]

    def __call__(self, values: pd.Series) -> pd.Series:
        def evaluate(x: ArrayLike):
            return threshold_operators[self.operator](x, self.threshold)

        return values.apply(evaluate)

    def validate_feature(self, feature: "AnyInput") -> None:
        if isinstance(feature, (CategoricalInput, MolecularInput)):
            raise TypeError(
                f"Feature {feature.key} is a {type(feature).__name__}, and cannot be used "
                f"with a {type(self).__name__}."
            )


class SelectionCondition(Condition):
    """Class for modelling selection conditions.

    It is checked if the feature value is in the selection of values. If this is the case,
    the condition is fulfilled. It can be only applied to CategoricalInput and DiscreteInput
    features.

    Attributes:
        selection: In case of CategoricalInput, the selection of categories to be included.
            In case of DiscreteInput, the selection of values to be included.
    """

    type: Literal["SelectionCondition"] = "SelectionCondition"

    selection: List[Union[str, float, int]]

    def __call__(self, values: pd.Series) -> pd.Series:
        return values.isin(self.selection)

    def validate_feature(self, feature: "AnyInput") -> None:
        if isinstance(feature, (ContinuousInput, MolecularInput)):
            raise TypeError(
                f"Feature {feature.key} is a {type(feature).__name__}, and cannot be used "
                f"with a {type(self).__name__}."
            )

        values = (
            feature.values if isinstance(feature, DiscreteInput) else feature.categories
        )
        if not all(val in values for val in self.selection):
            raise ValueError(
                f"Some categories in {type(self).__name__} are not in {feature.key}."
            )


class NonZeroCondition(Condition):
    """Class for modelling non-zero conditions.

    It is checked if the feature value is not equal to zero. If this is the case,
    the condition is fulfilled. It can be only applied to ContinuousInput and DiscreteInput
    features.
    """

    type: Literal["NonZeroCondition"] = "NonZeroCondition"

    def __call__(self, values: pd.Series) -> pd.Series:
        return values != 0

    def validate_feature(self, feature: "AnyInput") -> None:
        if isinstance(feature, (CategoricalInput, MolecularInput)):
            raise TypeError(
                f"Feature {feature.key} is a {type(feature).__name__}, and cannot be used "
                f"with a {type(self).__name__}."
            )
