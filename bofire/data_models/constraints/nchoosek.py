from typing import List, Literal

import numpy as np
import pandas as pd
from pydantic import root_validator, validator

from bofire.data_models.constraints.constraint import Constraint, FeatureKeys


def narrow_gaussian(x, ell=1e-3):
    return np.exp(-0.5 * (x / ell) ** 2)


class NChooseKConstraint(Constraint):
    """NChooseK constraint that defines how many ingredients are allowed in a formulation.

    Attributes:
        features (List[str]): List of feature keys to which the constraint applies.
        min_count (int): Minimal number of non-zero/active feature values.
        max_count (int): Maximum number of non-zero/active feature values.
        none_also_valid (bool): In case that min_count > 0,
            this flag decides if zero active features are also allowed.
    """

    type: Literal["NChooseKConstraint"] = "NChooseKConstraint"
    features: FeatureKeys
    min_count: int
    max_count: int
    none_also_valid: bool

    @validator("features")
    def validate_features_unique(cls, features: List[str]):
        """Validates that provided feature keys are unique."""
        if len(features) != len(set(features)):
            raise ValueError("features must be unique")
        return features

    @root_validator(pre=False, skip_on_failure=True)
    def validate_counts(cls, values):
        """Validates if the minimum and maximum of allowed features are smaller than the overall number of features."""
        features = values["features"]
        min_count = values["min_count"]
        max_count = values["max_count"]

        if min_count > len(features):
            raise ValueError("min_count must be <= # of features")
        if max_count > len(features):
            raise ValueError("max_count must be <= # of features")
        if min_count > max_count:
            raise ValueError("min_values must be <= max_values")

        return values

    def __call__(self, experiments: pd.DataFrame) -> pd.Series:
        """Smooth relaxation of NChooseK constraint by countig the number of zeros in a candidate by a sum of
        narrow gaussians centered at zero.

        Args:
            experiments (pd.DataFrame): Data to evaluate the constraint on.

        Returns:
            pd.Series containing the constraint violation for each experiment (row in experiments argument).
        """

        def relu(x):
            return np.maximum(0, x)

        indices = np.array(
            [i for i, name in enumerate(experiments.columns) if name in self.features],
            dtype=np.int64,
        )
        experiments_tensor = np.array(experiments.to_numpy())

        max_count_violation = np.zeros(experiments_tensor.shape[0])
        min_count_violation = np.zeros(experiments_tensor.shape[0])

        if self.max_count != len(self.features):
            max_count_violation = relu(-1 * narrow_gaussian(x=experiments_tensor[..., indices]).sum(axis=-1) + (len(self.features) - self.max_count))  # type: ignore

        if self.min_count > 0:
            min_count_violation = relu(narrow_gaussian(x=experiments_tensor[..., indices]).sum(axis=-1) - (len(self.features) - self.min_count))  # type: ignore

        return pd.Series(max_count_violation + min_count_violation)

    def is_fulfilled(self, experiments: pd.DataFrame, tol: float = 1e-6) -> pd.Series:
        """Check if the concurrency constraint is fulfilled for all the rows of the provided dataframe.

        Args:
            experiments (pd.DataFrame): Dataframe to evaluate constraint on.
            tol (float,optional): tolerance parameter. A constraint is considered as not fulfilled
                if the violation is larger than tol. Defaults to 1e-6.

        Returns:
            bool: True if fulfilled else False.
        """

        cols = self.features
        sums = (np.abs(experiments[cols]) > tol).sum(axis=1)

        lower = sums >= self.min_count
        upper = sums <= self.max_count

        if not self.none_also_valid:
            # return lower.all() and upper.all()
            return pd.Series(np.logical_and(lower, upper), index=experiments.index)
        else:
            none = sums == 0
            return pd.Series(
                np.logical_or(none, np.logical_and(lower, upper)),
                index=experiments.index,
            )

    def __str__(self):
        """Generate string representation of the constraint.

        Returns:
            str: string representation of the constraint.
        """
        res = (
            "of the features "
            + ", ".join(self.features)
            + f" between {self.min_count} and {self.max_count} must be used"
        )
        if self.none_also_valid:
            res += " (none is also ok)"
        return res

    def jacobian(self, experiments: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError("Jacobian not implemented for NChooseK constraints.")
