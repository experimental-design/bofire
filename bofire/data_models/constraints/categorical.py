import operator as ops
from abc import abstractmethod
from typing import Annotated, Any, Callable, Dict, List, Literal, Optional, Union

import pandas as pd
from numpy.typing import ArrayLike
from pydantic import Field

from bofire.data_models.base import BaseModel
from bofire.data_models.constraints.constraint import Constraint
from bofire.data_models.domain.features import Inputs
from bofire.data_models.features.api import (
    CategoricalInput,
    ContinuousInput,
    DiscreteInput,
)
from bofire.data_models.types import FeatureKeys


_threshold_operators: dict[str, Callable] = {
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
            return _threshold_operators[self.operator](x, self.threshold)

        return values.apply(evaluate)


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


class CategoricalExcludeConstraint(Constraint):
    """Class for modelling exclusion constraints.

    It evaluates conditions on two features and combines them using logical operators.
    If the logical combination evaluates to true, the constraint is not fulfilled.
    So far, this kind of constraint is only supported by the RandomStrategy.

    Attributes:
        features: List of feature keys to apply the conditions on.
        conditions: List of conditions to evaluate.
        logical_op: Logical operator to combine the conditions. Can be "AND", "OR", or "XOR".
            Default is "AND".
    """

    type: Literal["CategoricalExcludeConstraint"] = "CategoricalExcludeConstraint"
    features: FeatureKeys
    conditions: Annotated[
        List[Union[ThresholdCondition, SelectionCondition]],
        Field(min_length=2, max_length=2),
    ]
    logical_op: Literal["AND", "OR", "XOR"] = "AND"

    def validate_inputs(self, inputs: Inputs):
        """Validates that the features stored in Inputs are compatible with the constraint.

        Args:
            inputs: Inputs to validate.
        """
        found_categorical = False
        keys = inputs.get_keys([CategoricalInput, DiscreteInput, ContinuousInput])
        for f in self.features:
            if f not in keys:
                raise ValueError(
                    f"Feature {f} is not a input feature in the provided Inputs object.",
                )
        for i in range(2):
            feat = inputs.get_by_key(self.features[i])
            condition = self.conditions[i]
            if isinstance(feat, CategoricalInput):
                found_categorical = True
                if not isinstance(condition, SelectionCondition):
                    raise ValueError(
                        f"Condition for feature {self.features[i]} is not a SubSelectionCondition.",
                    )
                if not all(key in feat.categories for key in condition.selection):
                    raise ValueError(
                        f"Some categories in condition {i} are not valid categories for feature {self.features[i]}."
                    )
            elif isinstance(feat, DiscreteInput):
                if isinstance(condition, SelectionCondition):
                    if not all(key in feat.values for key in condition.selection):
                        raise ValueError(
                            f"Some values in condition {i} are not valid values for feature {self.features[i]}."
                        )
            else:  # we have a ContinuousInput
                if not isinstance(condition, ThresholdCondition):
                    raise ValueError(
                        f"Condition for ContinuousInput {self.features[i]} is not a ThresholdCondition.",
                    )
        if not found_categorical:
            raise ValueError(
                "At least one of the features must be a CategoricalInput feature.",
            )

    def __call__(self, experiments: pd.DataFrame) -> pd.Series:
        """Numerically evaluates the constraint.

        Returns the distance to the constraint fulfillment. Here 1 for not fulfilled
        and 0 for fulfilled.

        Args:
            experiments (pd.DataFrame): Dataframe to evaluate the constraint on.

        Returns:
            pd.Series: Distance to reach constraint fulfillment.

        """
        return self.is_fulfilled(experiments).astype(float)

    def is_fulfilled(
        self,
        experiments: pd.DataFrame,
        tol: Optional[float] = 1e-6,
    ) -> pd.Series:
        """Checks if the constraint is fulfilled for the given experiments.

        Args:
            experiments DataFrame containing the experiments.
            tol: Tolerance for checking. Not used here. Defaults to 1e-6.

        Returns:
            Series indicating whether the constraint is fulfilled for each experiment.

        """
        fulfilled_conditions = [
            condition(experiments[self.features[i]])
            for i, condition in enumerate(self.conditions)
        ]

        if self.logical_op == "AND":
            return ~(fulfilled_conditions[0] & fulfilled_conditions[1])
        elif self.logical_op == "OR":
            return ~(fulfilled_conditions[0] | fulfilled_conditions[1])
        else:
            return ~(fulfilled_conditions[0] ^ fulfilled_conditions[1])

    def jacobian(self, experiments: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError("Method `jacobian` currently not implemented.")

    def hessian(self, experiments: pd.DataFrame) -> Dict[Union[str, int], pd.DataFrame]:
        raise NotImplementedError("Method `hessian` currently not implemented.")
