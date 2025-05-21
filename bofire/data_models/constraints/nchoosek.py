from typing import Dict, Literal, Union

import numpy as np
import pandas as pd
from pydantic import model_validator

from bofire.data_models.constraints.constraint import IntrapointConstraint
from bofire.data_models.domain.features import Inputs
from bofire.data_models.features.api import ContinuousInput, DiscreteInput


def narrow_gaussian(x, ell=1e-3):
    return np.exp(-0.5 * (x / ell) ** 2)


class NChooseKConstraint(IntrapointConstraint):
    """NChooseK constraint that defines how many ingredients are allowed in a formulation.

    Attributes:
        features (List[str]): List of feature keys to which the constraint applies.
        min_count (int): Minimal number of non-zero/active feature values.
        max_count (int): Maximum number of non-zero/active feature values.
        none_also_valid (bool): In case that min_count > 0,
            this flag decides if zero active features are also allowed.

    """

    type: Literal["NChooseKConstraint"] = "NChooseKConstraint"
    min_count: int
    max_count: int
    none_also_valid: bool

    def validate_inputs(self, inputs: Inputs):
        keys = inputs.get_keys([ContinuousInput, DiscreteInput])
        for f in self.features:
            if f not in keys:
                raise ValueError(
                    f"Feature {f} is not a continuous input feature in the provided Inputs object.",
                )
            feature_ = inputs.get_by_key(f)
            assert isinstance(
                feature_, ContinuousInput
            ), f"Feature {f} is not a ContinuousInput."
            if feature_.bounds[0] < 0:
                raise ValueError(
                    f"Feature {f} must have a lower bound of >=0, but has {feature_.bounds[0]}",
                )

    @model_validator(mode="after")
    def validate_counts(self):
        """Validates if the minimum and maximum of allowed features are smaller than the overall number of features."""
        if self.min_count > len(self.features):
            raise ValueError("min_count must be <= # of features")
        if self.max_count > len(self.features):
            raise ValueError("max_count must be <= # of features")
        if self.min_count > self.max_count:
            raise ValueError("min_values must be <= max_values")

        return self

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
            max_count_violation = relu(
                -1 * narrow_gaussian(x=experiments_tensor[..., indices]).sum(axis=-1)
                + (len(self.features) - self.max_count),
            )

        if self.min_count > 0:
            min_count_violation = relu(
                narrow_gaussian(x=experiments_tensor[..., indices]).sum(axis=-1)
                - (len(self.features) - self.min_count),
            )

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
        none = sums == 0
        return pd.Series(
            np.logical_or(none, np.logical_and(lower, upper)),
            index=experiments.index,
        )

    def jacobian(self, experiments: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError("Jacobian not implemented for NChooseK constraints.")

    def hessian(self, experiments: pd.DataFrame) -> Dict[Union[str, int], pd.DataFrame]:
        raise NotImplementedError("Hessian not implemented for NChooseK constraints.")
