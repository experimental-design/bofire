from typing import ClassVar, Literal

import numpy as np
import pandas as pd
from pydantic import validator

from bofire.data_models.features.feature import TDiscreteVals
from bofire.data_models.features.numerical import NumericalInput


class DiscreteInput(NumericalInput):
    """Feature with discretized ordinal values allowed in the optimization.

    Attributes:
        key(str): key of the feature.
        values(List[float]): the discretized allowed values during the optimization.
    """

    type: Literal["DiscreteInput"] = "DiscreteInput"
    order_id: ClassVar[int] = 3

    values: TDiscreteVals

    @validator("values")
    def validate_values_unique(cls, values):
        """Validates that provided values are unique.

        Args:
            values (List[float]): List of values

        Raises:
            ValueError: when values are non-unique.
            ValueError: when values contains only one entry.
            ValueError: when values is empty.

        Returns:
            List[values]: Sorted list of values
        """
        if len(values) != len(set(values)):
            raise ValueError("Discrete values must be unique")
        if len(values) == 1:
            raise ValueError(
                "Fixed discrete inputs are not supported. Please use a fixed continuous input."
            )
        if len(values) == 0:
            raise ValueError("No values defined.")
        return sorted(values)

    @property
    def lower_bound(self) -> float:
        """Lower bound of the set of allowed values"""
        return min(self.values)

    @property
    def upper_bound(self) -> float:
        """Upper bound of the set of allowed values"""
        return max(self.values)

    def validate_candidental(self, values: pd.Series) -> pd.Series:
        """Method to validate the provided candidates.

        Args:
            values (pd.Series): suggested candidates for the feature

        Raises:
            ValueError: Raises error when one of the provided values is not contained in the list of allowed values.

        Returns:
            pd.Series: _uggested candidates for the feature
        """
        values = super().validate_candidental(values)
        if not np.isin(values.to_numpy(), np.array(self.values)).all():
            raise ValueError(
                f"Not allowed values in candidates for feature {self.key}."
            )
        return values

    def sample(self, n: int) -> pd.Series:
        """Draw random samples from the feature.

        Args:
            n (int): number of samples.

        Returns:
            pd.Series: drawn samples.
        """
        return pd.Series(name=self.key, data=np.random.choice(self.values, n))

    def from_continuous(self, values: pd.DataFrame) -> pd.Series:
        """Rounds continuous values to the closest discrete ones.

        Args:
            values (pd.DataFrame): Dataframe with continuous entries.

        Returns:
            pd.Series: Series with discrete values.
        """

        s = pd.DataFrame(
            data=np.abs(
                (values[self.key].to_numpy()[:, np.newaxis] - np.array(self.values))
            ),
            columns=self.values,
            index=values.index,
        ).idxmin(1)
        s.name = self.key
        return s
