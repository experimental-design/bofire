from typing import Annotated, Dict, List, Literal, Union

import numpy as np
import pandas as pd
from pydantic import AfterValidator, Field

from bofire.data_models.constraints.constraint import Constraint
from bofire.data_models.domain.features import Inputs
from bofire.data_models.features.api import CategoricalInput, DiscreteInput
from bofire.data_models.types import make_unique_validator

from .categorical import SelectionCondition, ThresholdCondition


FeatureKeysAnyLength = Annotated[
    List[str],
    Field(min_length=1),
    AfterValidator(make_unique_validator("Features")),
]


class ConditionallyActiveConstraint(Constraint):
    """Constraint features to only be active subject to conditions.

    Conditional (or hierarchical) features only impact the surrogate if a condition
    is met. This condition can depend on other features"""

    type: Literal["ConditionalConstraint"] = "ConditionalConstraint"
    features: FeatureKeysAnyLength
    conditions: list[tuple[str, ThresholdCondition | SelectionCondition]]
    logical_op: Literal["AND", "OR"] = "AND"

    def validate_inputs(self, inputs: Inputs):
        """Validates that the features stored in Inputs are compatible with the constraint.

        Args:
            inputs: Inputs to validate.
        """
        for f in self.features:
            feat = inputs.get_by_key(f)
            if isinstance(feat, (CategoricalInput, DiscreteInput)):
                raise ValueError(
                    f"The dependent features in the conditionally active constraint "
                    f"must be continuous. Got {f} of type {feat.__class__.__name__}.",
                )

        for f_cond, condition in self.conditions:
            feat = inputs.get_by_key(f_cond)
            if isinstance(feat, CategoricalInput):
                if not isinstance(condition, SelectionCondition):
                    raise ValueError(
                        f"Condition for feature {feat} is not a SubSelectionCondition.",
                    )
                if not all(key in feat.categories for key in condition.selection):
                    raise ValueError(
                        f"Some categories in {condition.__class__.__name__} are not valid categories for feature {f_cond}."
                    )
            elif isinstance(feat, DiscreteInput):
                if isinstance(condition, SelectionCondition):
                    if not all(key in feat.values for key in condition.selection):
                        raise ValueError(
                            f"Some values in condition {condition.__class__.__name__} are not valid values for feature {f_cond}."
                        )
            else:  # we have a ContinuousInput
                if not isinstance(condition, ThresholdCondition):
                    raise ValueError(
                        f"Condition for ContinuousInput {feat} is not a ThresholdCondition.",
                    )

    def __call__(self, experiments: pd.DataFrame) -> pd.Series:
        """Numerically evaluates the constraint.

        Since this is not a true constraint, it is always satisfied

        Args:
            experiments (pd.DataFrame): Dataframe to evaluate the constraint on.

        Returns:
            pd.Series: Distance to reach constraint fulfillment.

        """
        return pd.Series(
            data=np.zeros((experiments.shape[0],)), index=experiments.index
        )

    def jacobian(self, experiments: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError("Method `jacobian` currently not implemented.")

    def hessian(self, experiments: pd.DataFrame) -> Dict[Union[str, int], pd.DataFrame]:
        raise NotImplementedError("Method `hessian` currently not implemented.")
